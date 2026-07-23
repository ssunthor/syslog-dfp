# Author: Scott Chrestman
# Copyright: 2026 GTT Communications, Inc.

import os
import logging
import joblib
import pickle
import pandas as pd
import numpy as np
import warnings
import uuid
import shutil
import time as pytime
from concurrent.futures import ThreadPoolExecutor
import collections
import threading
import gc
import json
import psutil
from datetime import datetime, timedelta
import rmm
import rmm.mr as rmm_mr
from confluent_kafka import Producer, Consumer, KafkaError

# --- CUDA / ML ---
import cudf
import cupy as cp
import torch
import mlflow

# Initialize RMM with a fixed pool + failure callback so OOM attempts cache
# flushes and retries before raising, rather than crashing the pipeline.
_rmm_oom_count = 0
_rmm_oom_window_start = 0.0

kafka_batch_size = os.getenv("KAFKA_BATCH_SIZE", 1024)
kafka_edge_buffer = os.getenv("KAFKA_EDGE_BUFFER", 128)

def _rmm_oom_handler(nbytes, nothrow=False):
    global _rmm_oom_count, _rmm_oom_window_start
    now = pytime.time()
    if now - _rmm_oom_window_start > 10.0:
        _rmm_oom_count = 0
        _rmm_oom_window_start = now
    _rmm_oom_count += 1

    _log = logging.getLogger(__name__)
    if _rmm_oom_count == 1:
        _log.warning(f"RMM pool exhausted (requested {nbytes:,} B) — flushing caches and retrying")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return True
    if _rmm_oom_count <= 3:
        return True  # retry without re-flushing — nothing new to free
    _log.error(f"RMM pool exhausted (requested {nbytes:,} B) — {_rmm_oom_count} failures in 10s, raising")
    return False  # propagate the exception so the stage can recover

try:
    _rmm_max_gb = float(os.environ.get("RMM_POOL_MAX_GB", "3"))
    _rmm_init_gb = min(1.0, _rmm_max_gb)
    _pool_mr = rmm_mr.PoolMemoryResource(
        rmm_mr.CudaMemoryResource(),
        initial_pool_size=int(_rmm_init_gb * 1024 ** 3),
        maximum_pool_size=int(_rmm_max_gb * 1024 ** 3),
    )
    rmm_mr.set_current_device_resource(
        rmm_mr.FailureCallbackResourceAdaptor(_pool_mr, _rmm_oom_handler)
    )
    if torch.cuda.is_available():
        _pt_fraction = float(os.environ.get("PYTORCH_MEMORY_FRACTION", "0.3"))
        torch.cuda.set_per_process_memory_fraction(_pt_fraction, 0)
except Exception as e:
    print(f"⚠️ Memory Initialization Warning: {e}")

# Morpheus
import mrc
from mrc.core import operators as ops
from morpheus.config import Config, ExecutionMode, ConfigAutoEncoder
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.messages import ControlMessage, MessageMeta
from morpheus.cli.register_stage import register_stage
from morpheus.utils.column_info import ColumnInfo

# Stages
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage as KafkaSource
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage

# DFP Model Import
try:
    from morpheus.models.dfencoder import AutoEncoder
except ImportError:
    from morpheus_dfp.models.autoencoder import AutoEncoder

# FastAPI for Health Checks
from fastapi import FastAPI, Response, status
from prometheus_client import make_asgi_app, Counter, Histogram
import uvicorn

PIPELINE_EVENTS_PROCESSED = Counter(
    'morpheus_pipeline_events_processed_total', 
    'Total events processed by the Morpheus pipeline',
    ['pipeline_type']
)

PIPELINE_ANOMALY_LOSS = Histogram(
    'morpheus_pipeline_anomaly_loss',
    'Distribution of AutoEncoder reconstruction loss (RawLoss)',
    ['pipeline_type'],
    buckets=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf"))
)

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize metrics so they appear in output even before data is processed
PIPELINE_EVENTS_PROCESSED.labels(pipeline_type="syslog").inc(0)
PIPELINE_ANOMALY_LOSS.labels(pipeline_type="syslog")
PIPELINE_HEALTH = {
    "status": "starting",
    "last_activity": pytime.time(),
    "is_running": False
}

@app.get("/health")
def health_check(response: Response):
    now = pytime.time()
    last_act = PIPELINE_HEALTH["last_activity"]
    is_running = PIPELINE_HEALTH["is_running"]
    if not is_running:
        if (now - last_act) > 300:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "unhealthy", "reason": "Pipeline failed to start within timeout"}
        return {"status": "starting", "pipeline_status": PIPELINE_HEALTH["status"]}
    if (now - last_act) > 600:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "reason": "Pipeline stalled (no activity for 10m)"}
    return {"status": "healthy", "pipeline_status": PIPELINE_HEALTH["status"]}

def run_health_server():
    logger.info("📡 Health Server starting on port 13005...")
    uvicorn.run(app, host="0.0.0.0", port=13005, log_level="warning")

# --- LOGGING & GLOBALS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("morpheus_inference")
warnings.filterwarnings("ignore")
logging.getLogger("autoencoder").setLevel(logging.CRITICAL)
logging.getLogger("morpheus.models.dfencoder").setLevel(logging.CRITICAL)

MLFLOW_LOCK = threading.Lock()
MODEL_DIR = "/app/models/dfp-models"
FEEDBACK_DIR = os.path.join(MODEL_DIR, "feedback")
if not os.path.exists(FEEDBACK_DIR): os.makedirs(FEEDBACK_DIR, exist_ok=True)

TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=4) 
TRAINING_IN_PROGRESS = set()
TRAINING_SET_LOCK = threading.Lock()
MODEL_UPDATED_EVENT = threading.Event()
TRAINING_LOCK = threading.Lock()

# --- DFP FEATURE DEFINITIONS (Exact Order from 78861f4d) ---
NUM_LOG_FEATURES = 8
FEATURE_COLS = ["feature_log_count", "feature_priority", "feature_event_type", "feature_is_security_event"]
FEATURE_COLS += ["feature_hour_sin", "feature_hour_cos", "feature_day_sin", "feature_day_cos"]
FEATURE_COLS += ["feature_user_hash", "feature_msg_len", "feature_proc_hash"]
for i in range(NUM_LOG_FEATURES):
    FEATURE_COLS.append(f"feature_log_v{i}")
NUM_FEATURES = len(FEATURE_COLS)

# Shared model cache between training and inference
GLOBAL_MODEL_CACHE = {}
GLOBAL_TRAIN_COUNTER = {}
MODEL_CACHE_LOCK = threading.Lock()

# Per-host rolling window of recent feature vectors used for DFP training.
# Without this, each training run only sees the current 2-second batch (100-16k rows),
# which is far too little data to learn stable behavioral baselines.
# DFP_ROLLING_WINDOW_ROWS controls the cap per host (default: 50k rows).
# At 100 EPS per host that is ~8 minutes of history; tune upward for busier hosts.
ROLLING_WINDOW_CACHE = {}   # {user_id: pd.DataFrame}
ROLLING_WINDOW_LOCK = threading.Lock()
ROLLING_WINDOW_MAX_ROWS = int(os.getenv("DFP_ROLLING_WINDOW_ROWS", "50000"))

CONTROL_TOPIC = os.getenv("KAFKA_CONTROL_TOPIC", "morpheus-control")

def process_feedback_locally(source, features):
    """Internal logic to save feedback and fine-tune model if loaded."""
    if not source or not features: return None
    
    # Ensure all 19 feature columns are present; missing keys default to 0.0 to prevent
    # NaN values when this dict is later concatenated with the full training DataFrame.
    filtered_features = {k: float(features.get(k, 0.0)) for k in FEATURE_COLS}
    
    feedback_path = os.path.join(FEEDBACK_DIR, f"{source}.json")
    try:
        current_feedback = []
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, "r") as f:
                    current_feedback = json.load(f)
                if not isinstance(current_feedback, list):
                    current_feedback = []
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️ Corrupted feedback file for {source}, resetting: {e}")
                current_feedback = []
        
        # Keep last 100 feedback samples
        current_feedback.append(filtered_features)
        current_feedback = current_feedback[-100:]
        
        # ATOMIC WRITE: Write to temp file then move to prevent "two JSON strings" corruption
        temp_path = feedback_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(current_feedback, f)
        os.replace(temp_path, feedback_path)
        
        logger.info(f"📝 FEEDBACK [Local]: Processed for {source}. Total overrides: {len(current_feedback)}")
        
        # Immediate Fine-tune if model is in cache
        with MODEL_CACHE_LOCK:
            model = GLOBAL_MODEL_CACHE.get(source)
            if model:
                def fine_tune():
                    try:
                        # AGGRESSIVE NUDGE: Convert to DataFrame
                        df = pd.DataFrame([filtered_features])
                        # Upsample feedback significantly to ensure it shifts the weights
                        df_boosted = pd.concat([df] * 100, ignore_index=True)
                        
                        with TRAINING_LOCK:
                            # Train for multiple epochs on the nudge data
                            model.fit(df_boosted, epochs=3)
                            
                            # CRITICAL: We must also update the model's normalization buffers 
                            # (saved_means/saved_scales) so the score actually drops.
                            if hasattr(model, "saved_means"):
                                device = "cuda" if torch.cuda.is_available() else "cpu"
                                feedback_tensor = torch.as_tensor(df.values[0], dtype=torch.float32, device=device)
                                
                                # Mix in the new "normal" mean (20% weight to the nudge)
                                current_means = model.saved_means
                                new_means = (current_means * 0.8) + (feedback_tensor * 0.2)
                                model.register_buffer("saved_means", new_means)
                                
                                # Also slightly widen the scales for the contributing features
                                # to reduce sensitivity for this specific pattern
                                current_scales = model.saved_scales
                                new_scales = torch.maximum(current_scales, torch.abs(current_means - feedback_tensor) * 1.5)
                                # Maintain the floor of 2.0
                                new_scales = torch.maximum(new_scales, torch.tensor(2.0, device=device))
                                model.register_buffer("saved_scales", new_scales)

                        # Signal reload so inference engine picks up the new weights
                        MODEL_UPDATED_EVENT.set()
                        logger.info(f"🎯 FINE-TUNE [Local]: Applied AGGRESSIVE nudge for {source}")
                    except Exception as e:
                        logger.error(f"Fine-tune error: {e}")
                
                TRAINING_EXECUTOR.submit(fine_tune)
        return len(current_feedback)
    except Exception as e:
        logger.error(f"Local feedback processing error: {e}")
        return None

def control_worker():
    """Background thread to listen for control messages (Feedback) from Kafka."""
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    # EVERY pod needs a unique group ID to ensure they all get the broadcast
    pod_id = f"morpheus-control-{uuid.uuid4().hex[:8]}"
    
    conf = {
        'bootstrap.servers': broker,
        'group.id': pod_id,
        'auto.offset.reset': 'latest',
        # Control worker can be starved of CPU during heavy GPU batch processing;
        # raise the interval so it doesn't get kicked from the group under load.
        'max.poll.interval.ms': '1800000',  # 30 minutes
    }
    
    consumer = Consumer(conf)
    consumer.subscribe([CONTROL_TOPIC])
    
    logger.info(f"📡 Control Worker started. Subscribed to {CONTROL_TOPIC} (Group: {pod_id})")
    
    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logger.error(f"Control Kafka Error: {msg.error()}")
            continue
            
        try:
            data = json.loads(msg.value().decode('utf-8'))
            if data.get("type") == "feedback":
                source = data.get("source")
                features = data.get("features")
                logger.info(f"📩 Received Broadcast Feedback for {source}")
                process_feedback_locally(source, features)
        except Exception as e:
            logger.error(f"Control message processing error: {e}")

@app.post("/feedback")
async def add_feedback(data: dict):
    """
    Receives feedback from the UI to mark a log entry as 'normal'.
    (Kept for backward compatibility, but backend now broadcasts via Kafka)
    """
    source = data.get("source")
    features = data.get("features")
    overrides = process_feedback_locally(source, features)
    if overrides is not None:
        return {"status": "success", "overrides": overrides}
    return {"status": "error", "message": "Failed to process feedback"}

def warm_up_model_cache():
    if not os.path.exists(MODEL_DIR): return
    logger.info(f"🕵️ WARMING UP: Scanning {MODEL_DIR} for existing models...")
    count = 0
    try:
        for host_dir_name in os.listdir(MODEL_DIR):
            if not host_dir_name.startswith("dfp-"): continue
            user_id = host_dir_name.replace("dfp-", "")
            if user_id in GLOBAL_MODEL_CACHE: continue
            
            host_path = os.path.join(MODEL_DIR, host_dir_name)
            try:
                # Look for model-v*.pt files
                model_files = [f for f in os.listdir(host_path) if f.startswith("model-v") and f.endswith(".pt")]
                if not model_files: continue
                
                # Get latest version
                latest_model_file = sorted(model_files, key=lambda x: int(x.split('-v')[1].split('.')[0]))[-1]
                latest_v_num = int(latest_model_file.split('-v')[1].split('.')[0])
                model_path = os.path.join(host_path, latest_model_file)
                
                if os.path.exists(model_path):
                    # Use raw torch.load instead of mlflow
                    loaded = torch.load(model_path)
                    if torch.cuda.is_available(): 
                        loaded.to("cuda").float()
                    
                    GLOBAL_MODEL_CACHE[user_id] = loaded
                    GLOBAL_TRAIN_COUNTER[user_id] = latest_v_num
                    count += 1

                    # Restore rolling window snapshot so the host immediately
                    # re-qualifies for training without re-accumulating from zero.
                    snapshot_path = os.path.join(host_path, "window_snapshot.pkl")
                    if os.path.exists(snapshot_path):
                        try:
                            with open(snapshot_path, "rb") as sf:
                                snap_df = pickle.load(sf)
                            if isinstance(snap_df, pd.DataFrame) and len(snap_df) > 0:
                                with ROLLING_WINDOW_LOCK:
                                    ROLLING_WINDOW_CACHE[user_id] = snap_df
                        except Exception as snap_err:
                            logger.warning(f"Could not restore window snapshot for {user_id}: {snap_err}")
            except Exception as e:
                logger.warning(f"Error loading model for {user_id}: {e}")
    except Exception as e:
        logger.error(f"Warm up error: {e}")
    
    logger.info(f"🚀 WARM UP COMPLETE: Loaded {count} models into memory.")

def memory_manager_worker():
    while True:
        try:
            cpu_pct = psutil.virtual_memory().percent
            gpu_pct = 0.0
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                gpu_pct = (1.0 - free / total) * 100.0
            if cpu_pct > 60 or gpu_pct > 85:
                if gpu_pct > 85:
                    logging.getLogger(__name__).warning(
                        f"GPU memory pressure: {gpu_pct:.1f}% used — freeing caches"
                    )
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
            pytime.sleep(10)
        except: pytime.sleep(30)

def model_janitor_worker():
    """Deletes models for hosts that haven't been seen in 48 hours to save disk space."""
    TTL_SECONDS = 48 * 3600
    while True:
        try:
            if os.path.exists(MODEL_DIR):
                now = pytime.time()
                for host_dir_name in os.listdir(MODEL_DIR):
                    if not host_dir_name.startswith("dfp-"): continue
                    host_path = os.path.join(MODEL_DIR, host_dir_name)
                    
                    # Check the mtime of the directory or the model file
                    mtime = os.path.getmtime(host_path)
                    if (now - mtime) > TTL_SECONDS:
                        logger.info(f"🧹 JANITOR: Pruning stale host model: {host_dir_name} (Last seen > 48h ago)")
                        shutil.rmtree(host_path)
                        
                        # Also remove from in-memory caches
                        user_id = host_dir_name.replace("dfp-", "")
                        with MODEL_CACHE_LOCK:
                            if user_id in GLOBAL_MODEL_CACHE:
                                del GLOBAL_MODEL_CACHE[user_id]
                                if user_id in GLOBAL_TRAIN_COUNTER:
                                    del GLOBAL_TRAIN_COUNTER[user_id]
                        with ROLLING_WINDOW_LOCK:
                            if user_id in ROLLING_WINDOW_CACHE:
                                del ROLLING_WINDOW_CACHE[user_id]
            
            # Run every 6 hours
            pytime.sleep(6 * 3600)
        except Exception as e:
            logger.error(f"Janitor error: {e}")
            pytime.sleep(3600)

# --- STAGES ---

@register_stage("accumulator")
class AccumulatorStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._buffer = []; self._count = 0; self._last_flush = pytime.time()
        # Tune batching: Wait up to 2 seconds or 16k logs
        self.batch_size = 16384; self.max_latency = 2.0; self.max_buffer_size = 32768 
        self._perf_last_time = pytime.time()
        self._perf_window_count = 0
    @property
    def name(self): return "accumulator"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(ControlMessage); return schema
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        PIPELINE_HEALTH["last_activity"] = pytime.time(); PIPELINE_HEALTH["is_running"] = True; PIPELINE_HEALTH["status"] = "processing"
        gdf = msg.payload().df; self._buffer.append(gdf); self._count += len(gdf); self._perf_window_count += len(gdf)
        PIPELINE_EVENTS_PROCESSED.labels(pipeline_type="syslog").inc(len(gdf))
        now = pytime.time()
        if self._count < self.batch_size and (now - self._last_flush < self.max_latency) and self._count < self.max_buffer_size: return None 
        try:
            big_gdf = cudf.concat(self._buffer, ignore_index=True); self._buffer = []; self._count = 0; self._last_flush = now
            dt = now - self._perf_last_time
            if dt >= 5.0:
                # Log the total windowed throughput to accurately reflect pod performance
                logger.info(f"📊 ACCUMULATOR: Window EPS: {self._perf_window_count/dt:.1f} | Last Flush: {big_gdf.shape[0]}")
                self._perf_last_time = now; self._perf_window_count = 0
            out = ControlMessage(); out.payload(MessageMeta(big_gdf)); return out
        except Exception as e: logger.error(f"Accumulator Flush Error: {e}"); self._buffer = []; self._count = 0; return None
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("gpu-preprocessing")
class GPUPreprocessingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._ip_regex = r"from\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        self._host_regex = r"(?:<\d+>)?([A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+([a-zA-Z0-9\.\-_]+)"
        self._user_regex = r"(?:user|for user|User|User=)\s*([a-zA-Z0-9\.\-_]+)"
        self._process_regex = r"(?:<\d+>)?(?:[A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+[a-zA-Z0-9\.\-_]+\s+([a-zA-Z0-9\.\-_]+)(?:\[\d+\])?: "
        # RFC 5424: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME ...
        # PRI + version digits distinguish 5424 from 3164 (which has a month name here);
        # \S+ consumes the whole timestamp token (fractional secs / TZ and all) up to the
        # space before HOSTNAME. The hostname group requires a leading alphanumeric so the
        # RFC NILVALUE ("-") is treated as "no hostname" rather than captured.
        self._rfc5424_host_regex = r"(?:<\d+>)\d+\s+\S+\s+([A-Za-z0-9][A-Za-z0-9\.\-_]*)"
        self._rfc5424_ts_regex = r"(?:<\d+>)\d+\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
        # FortiOS key=value logs (not RFC 3164/5424): the origin host is devname, the
        # stable serial is devid, and eventtime is a Unix-epoch (seconds) event time.
        # "?  makes the surrounding quotes optional so quoted and unquoted values both match.
        self._forti_devname_regex = r'devname="?([A-Za-z0-9\.\-_]+)'
        self._forti_devid_regex = r'devid="?([A-Za-z0-9\.\-_]+)'
        self._forti_eventtime_regex = r'eventtime=(\d+)'
    @property
    def name(self): return "gpu-preprocessing"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                for col in FEATURE_COLS: df[col] = 0.0
                now = datetime.utcnow()
                if "message" in df.columns:
                    msg_col = df["message"]
                    # Host identity, in priority order:
                    #   1. FortiOS devname / devid (key=value logs — no positional hostname)
                    #   2. RFC 3164 hostname (group 2, after the "Mmm dd hh:mm:ss" timestamp)
                    #   3. RFC 5424 HOSTNAME field
                    #   4. in-message "from <IP>" — usually a *peer* address, not the origin
                    # Preferring the device/host fields over the peer IP keeps a host on one
                    # stable identity instead of splitting across every remote IP in its logs.
                    extracted_devname = msg_col.str.extract(self._forti_devname_regex, expand=True)[0]
                    extracted_devid = msg_col.str.extract(self._forti_devid_regex, expand=True)[0]
                    extracted_host = msg_col.str.extract(self._host_regex, expand=True)[1]
                    extracted_host_5424 = msg_col.str.extract(self._rfc5424_host_regex, expand=True)[0]
                    extracted_ip = msg_col.str.extract(self._ip_regex, expand=True)[0]
                    df["source"] = (
                        extracted_devname
                        .fillna(extracted_devid)
                        .fillna(extracted_host)
                        .fillna(extracted_host_5424)
                        .fillna(extracted_ip)
                        .fillna("unknown_source")
                    )
                    df["feature_msg_len"] = (msg_col.str.len() / 512.0).astype("float32").clip(0.0, 1.0)
                    users = msg_col.str.extract(self._user_regex, expand=True)[0].fillna("none")
                    procs = msg_col.str.extract(self._process_regex, expand=True)[0].fillna("unknown")
                    # Extract to CuPy arrays before arithmetic to bypass cuDF index alignment.
                    # cp.abs() normalises negative int32 hash values before the modulo.
                    df["feature_user_hash"] = (cp.abs(users.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
                    df["feature_proc_hash"] = (cp.abs(procs.hash_values().values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)
                    for i in range(NUM_LOG_FEATURES):
                        df[f"feature_log_v{i}"] = (cp.abs(msg_col.hash_values(seed=i).values) % cp.int32(100000)).astype(cp.float32) / cp.float32(100000.0)

                    # Parse actual syslog timestamps so temporal features reflect real event time.
                    # Use pandas on the CPU: pd.to_datetime fully supports errors='coerce' for
                    # Series, avoiding multiple cudf.to_datetime quirks in v25.x (no Series-level
                    # errors='coerce', index-alignment issues with string + operator).
                    # The CPU round-trip is one string column (~1-2 MB at 16k rows) — negligible.
                    msg_pd = msg_col.to_pandas()
                    year_str = str(now.year) + ' '

                    # RFC 3164: "Jan  1 12:00:00" (group 0 of _host_regex)
                    ts_raw_pd = msg_pd.str.extract(self._host_regex, expand=True)[0]
                    ts_norm_pd = ts_raw_pd.str.replace(r'\s+', ' ', regex=True).str.strip()
                    parsed_3164_pd = pd.to_datetime(
                        year_str + ts_norm_pd, format="%Y %b %d %H:%M:%S", errors='coerce'
                    )

                    # RFC 5424: "2026-01-01T12:00:00..." (group 0 of _rfc5424_ts_regex)
                    ts_5424_pd = msg_pd.str.extract(self._rfc5424_ts_regex, expand=True)[0]
                    parsed_5424_pd = pd.to_datetime(
                        ts_5424_pd, format="%Y-%m-%dT%H:%M:%S", errors='coerce'
                    )

                    # FortiOS: "eventtime=1784823567" — Unix epoch seconds, timezone-unambiguous.
                    ts_epoch_pd = msg_pd.str.extract(self._forti_eventtime_regex, expand=True)[0]
                    parsed_epoch_pd = pd.to_datetime(
                        pd.to_numeric(ts_epoch_pd, errors='coerce'), unit='s', errors='coerce'
                    )

                    combined_ts = (
                        parsed_3164_pd
                        .fillna(parsed_5424_pd)
                        .fillna(parsed_epoch_pd)
                        .fillna(pd.Timestamp(now))
                    )
                    # Assign via numpy array (no pandas index) so cuDF assigns by position
                    df["timestamp"] = cudf.Series(combined_ts.values.astype("datetime64[ns]"))
                else:
                    df["source"] = "unknown_source"
                    df["timestamp"] = pd.Timestamp(now)

                # Per-row time features — extract to CuPy to bypass cuDF index alignment.
                hours_cp = (df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0).values
                days_cp = df["timestamp"].dt.weekday.astype("float32").values
                df["feature_hour_sin"] = cp.sin(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
                df["feature_hour_cos"] = cp.cos(2 * cp.float32(cp.pi) * hours_cp / cp.float32(24.0)).astype(cp.float32)
                df["feature_day_sin"] = cp.sin(2 * cp.float32(cp.pi) * days_cp / cp.float32(7.0)).astype(cp.float32)
                df["feature_day_cos"] = cp.cos(2 * cp.float32(cp.pi) * days_cp / cp.float32(7.0)).astype(cp.float32)
            except Exception as e: logger.error(f"Preprocessing Early Error: {e}")
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("gpu-feature-mapping")
class GPUFeatureMappingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        # Load class counts from the Triton label encoders so we normalize priority and
        # event_type by the actual number of classes rather than hardcoded guesses.
        self._priority_classes = 8    # RFC 3164 has 8 severity levels (0-7); safe default
        self._event_type_classes = 10  # fallback; overridden below if encoder loads
        model_dir = os.environ.get("INFERENCE_MODELS_DIR", "/app/models/inference")
        try:
            for attr, filename in [("_priority_classes", "priority_label_encoder.joblib"),
                                    ("_event_type_classes", "event_type_label_encoder.joblib")]:
                path = os.path.join(model_dir, filename)
                if os.path.exists(path):
                    enc = joblib.load(path)
                    if hasattr(enc, "classes_") and len(enc.classes_) > 1:
                        setattr(self, attr, len(enc.classes_))
                        logger.info(f"Loaded {filename}: {len(enc.classes_)} classes")
        except Exception as e:
            logger.warning(f"Could not load label encoders for feature normalization, using defaults: {e}")
    @property
    def name(self): return "gpu-feature-mapping"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                # Use the actual timestamp range of the batch to compute a real EPS.
                # Previously this used (rows * 1µs) as the window, producing values ~100x too high.
                if "timestamp" in df.columns:
                    # cuDF max/min on datetime returns numpy.timedelta64 (not pandas.Timedelta),
                    # so use pd.Timedelta() conversion to get total_seconds() reliably.
                    ts_diff = pd.Timedelta(df["timestamp"].max() - df["timestamp"].min())
                    duration_sec = max(ts_diff.total_seconds(), 1.0)  # floor at 1s for same-second batches
                else:
                    duration_sec = 2.0  # AccumulatorStage max_latency fallback
                counts = df['source'].value_counts()

                # OPTIMIZATION: Log-scaled volume (Prevents volume spikes from exploding the score)
                eps = (df['source'].map(counts) / duration_sec * 60.0).astype('float32')
                df['feature_log_count'] = cp.log1p(eps) / 10.0 # Normalized roughly for 0-20k EPS
                
                # Normalize Triton class labels to [0, 1] using the actual number of classes.
                # Dividing by (num_classes - 1) maps label 0 -> 0.0 and label N-1 -> 1.0.
                if "priority" in df.columns:
                    df["feature_priority"] = df["priority"].astype("float32") / float(self._priority_classes - 1)
                if "event_type" in df.columns:
                    df["feature_event_type"] = df["event_type"].astype("float32") / float(self._event_type_classes - 1)
                if "is_security_event" in df.columns:
                    df["feature_is_security_event"] = df["is_security_event"].astype("float32")
                
                df["anomaly_score"] = -1.0; df["mean_abs_err"] = -1.0; df["anomaly_cause"] = "analysis_pending" 
            except Exception as e: logger.error(f"Preprocessing Late Error: {e}")
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("async-training")
class AsyncTrainingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c):
        super().__init__(c); self._train_counter = {}; self._last_train_time = {} 
        self._save_interval = 50; self._queue_semaphore = threading.Semaphore(64) 
    @property
    def name(self): return "async-training"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _train_and_save(self, window_snapshot, user_id, count):
        try:
            train_data = window_snapshot  # already FEATURE_COLS only (from ROLLING_WINDOW_CACHE)
            
            # INCLUDE FEEDBACK: Merge saved user overrides into the training set
            feedback_path = os.path.join(FEEDBACK_DIR, f"{user_id}.json")
            if os.path.exists(feedback_path):
                try:
                    with open(feedback_path, "r") as f:
                        feedback_list = json.load(f)
                    if feedback_list and isinstance(feedback_list, list):
                        feedback_df = pd.DataFrame(feedback_list)
                        # Fill any columns missing from the feedback payload with 0.0 so the
                        # concat below never introduces NaN values into the training matrix.
                        for col in FEATURE_COLS:
                            if col not in feedback_df.columns:
                                feedback_df[col] = 0.0
                        feedback_df = feedback_df[FEATURE_COLS]
                        # Upsample feedback so it carries weight in the full training batch
                        feedback_upsampled = pd.concat([feedback_df] * 10, ignore_index=True)
                        train_data = pd.concat([train_data, feedback_upsampled], ignore_index=True)
                        logger.info(f"🧬 AUGMENTED: Added {len(feedback_list)} feedback samples (upsampled to {len(feedback_upsampled)}) to training for {user_id}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"⚠️ Training: Skipping corrupted feedback for {user_id}: {e}")
                except Exception as e:
                    logger.error(f"Error loading feedback for {user_id}: {e}")

            with MODEL_CACHE_LOCK:
                if user_id not in GLOBAL_MODEL_CACHE:
                    # PROPER BOTTLENECK: [12, 8] for 19 features ensures useful compression
                    GLOBAL_MODEL_CACHE[user_id] = AutoEncoder(encoder_layers=[12, 8], decoder_layers=[8, 12], activation='relu', patience=None)
                    if torch.cuda.is_available(): GLOBAL_MODEL_CACHE[user_id].to("cuda").float()
            model = GLOBAL_MODEL_CACHE.get(user_id)
            if not model: return

            # CALCULATE NORMALIZATION BEFORE TRAINING
            data_matrix = train_data.values
            if hasattr(data_matrix, 'get'): data_matrix = data_matrix.get()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            means = np.mean(data_matrix, axis=0)
            stds = np.std(data_matrix, axis=0)
            # SENSITIVE NORMALIZATION: Floor of 0.1 is more appropriate for 0-1 features
            robust_stds = np.maximum(stds, 0.1)
            
            # Normalize the training data so model learns on scaled input (MATCHES INFERENCE)
            # Using pandas broadcasting for consistency with FEATURE_COLS
            normalized_train_data = (train_data - means) / robust_stds
            
            with TRAINING_LOCK:
                # Train for 3 epochs for better convergence
                model.fit(normalized_train_data, epochs=3)

            # Register normalization buffers on the model BEFORE signaling the inference engine.
            # Previously MODEL_UPDATED_EVENT was set first, creating a race where _build_gpu_lookups
            # could read model.saved_means before register_buffer had run, loading stale stats.
            model.register_buffer("saved_means", torch.tensor(means, dtype=torch.float32, device=device))
            model.register_buffer("saved_scales", torch.tensor(robust_stds, dtype=torch.float32, device=device))

            # SYNC: Update global counter, then signal inference engine
            with MODEL_CACHE_LOCK:
                GLOBAL_TRAIN_COUNTER[user_id] = max(GLOBAL_TRAIN_COUNTER.get(user_id, 0), count)

            MODEL_UPDATED_EVENT.set()
            
            if count == 1 or (count > 0 and count % self._save_interval == 0):
                host_dir = os.path.join(MODEL_DIR, f"dfp-{user_id}")
                if count == 1: shutil.rmtree(host_dir, ignore_errors=True)
                os.makedirs(host_dir, exist_ok=True)
                
                # LIGHTWEIGHT SAVE: (~200KB vs ~15MB MLflow overhead)
                model_save_path = os.path.join(host_dir, f"model-v{count}.pt")
                torch.save(model, model_save_path)
                
                # VERSION PRUNING: Only keep the latest version on disk
                try:
                    for f in os.listdir(host_dir):
                        if f.startswith("model-v") and f != f"model-v{count}.pt":
                            os.remove(os.path.join(host_dir, f))
                except Exception as e:
                    logger.warning(f"Version pruning error for {user_id}: {e}")

                # ROLLING WINDOW SNAPSHOT: Persist the last 1 000 rows so the
                # host immediately re-qualifies for training after a restart.
                try:
                    with ROLLING_WINDOW_LOCK:
                        snap = ROLLING_WINDOW_CACHE.get(user_id)
                    if snap is not None and len(snap) > 0:
                        snap_tail = snap.tail(1000).reset_index(drop=True)
                        snap_path = os.path.join(host_dir, "window_snapshot.pkl")
                        with open(snap_path, "wb") as sf:
                            pickle.dump(snap_tail, sf, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as snap_err:
                    logger.warning(f"Could not save window snapshot for {user_id}: {snap_err}")

                MODEL_UPDATED_EVENT.set(); logger.info(f"💾 SAVED: Host {user_id} Version {count} (Lightweight)")
        except Exception as e: logger.error(f"❌ Training Thread Error for {user_id}: {e}")
        finally:
             self._queue_semaphore.release()
             with TRAINING_SET_LOCK:
                 if user_id in TRAINING_IN_PROGRESS: TRAINING_IN_PROGRESS.remove(user_id)
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        df = msg.payload().df; source_col = "source" if "source" in df.columns else "hostname" if "hostname" in df.columns else "host" if "host" in df.columns else None
        if not source_col: return msg
        # Only train on rows Triton successfully classified; rows with Triton down have
        # all Triton-derived features zeroed out and would corrupt the model baseline.
        train_df = df[df["triton_classified"] == True] if "triton_classified" in df.columns else df
        if len(train_df) == 0: return msg

        # Append this batch's classified feature vectors to each host's rolling window.
        # Training then uses the full window, not just the current 2-second batch.
        batch_pdf = train_df[[source_col] + FEATURE_COLS].to_pandas()
        with ROLLING_WINDOW_LOCK:
            for uid, group in batch_pdf.groupby(source_col):
                new_rows = group[FEATURE_COLS]
                if uid in ROLLING_WINDOW_CACHE:
                    ROLLING_WINDOW_CACHE[uid] = pd.concat(
                        [ROLLING_WINDOW_CACHE[uid], new_rows], ignore_index=True
                    ).tail(ROLLING_WINDOW_MAX_ROWS)
                else:
                    ROLLING_WINDOW_CACHE[uid] = new_rows.reset_index(drop=True)

        counts = train_df[source_col].value_counts()
        # Require at least 100 rows in the rolling window (not just this batch) before training
        with ROLLING_WINDOW_LOCK:
            eligible = [uid for uid in ROLLING_WINDOW_CACHE if len(ROLLING_WINDOW_CACHE[uid]) >= 100]
        if not eligible: return msg

        for user_id in eligible:
            if pytime.time() - self._last_train_time.get(user_id, 0) < 60: continue
            with TRAINING_SET_LOCK:
                if user_id in TRAINING_IN_PROGRESS: continue
            if not self._queue_semaphore.acquire(blocking=False): continue

            with MODEL_CACHE_LOCK:
                self._train_counter[user_id] = self._train_counter.get(user_id, GLOBAL_TRAIN_COUNTER.get(user_id, 0)) + 1
            count = self._train_counter[user_id]

            with ROLLING_WINDOW_LOCK:
                window_snapshot = ROLLING_WINDOW_CACHE[user_id].copy()

            logger.info(f"⚡ TRAINING START: Host {user_id} (window={len(window_snapshot)}) - V{count}")
            self._last_train_time[user_id] = pytime.time()
            with TRAINING_SET_LOCK:
                TRAINING_IN_PROGRESS.add(user_id)
            TRAINING_EXECUTOR.submit(self._train_and_save, window_snapshot, user_id, count)
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("manual-inference")
class ManualInferenceStage(SinglePortStage):
    # Time features (indices 4-7) are excluded from scoring and reason attribution.
    # The autoencoder cannot learn to reconstruct sin/cos time encoding — it trains
    # on logs from many hours so it learns the mean, which is wrong for every specific
    # hour. Time features stay in the training vector (the bottleneck still captures
    # time-correlated behavior) but contribute zero to scores and reasons.
    TIME_FEATURE_INDICES = {4, 5, 6, 7}
    FEATURE_WEIGHTS = [1.0] * NUM_FEATURES
    for _ti in TIME_FEATURE_INDICES:
        FEATURE_WEIGHTS[_ti] = 0.0

    def __init__(self, c):
        super().__init__(c); self._debug_counter = 0; self._gc_anchor = collections.deque(maxlen=5)
        self._last_reload = pytime.time(); self._min_reload_cooldown = 30.0
        self._feature_weights = torch.tensor(self.FEATURE_WEIGHTS, device="cuda", dtype=torch.float32)
        self._build_gpu_lookups()
    def _build_gpu_lookups(self):
        logger.info(f"⏳ RELOAD: Syncing GPU from Global Cache...")
        fallback_mean = [0.5] * NUM_FEATURES; fallback_std = [0.1] * NUM_FEATURES
        self.sources = ["UNKNOWN_FALLBACK"]; means = [fallback_mean]; stds = [fallback_std]; versions = [0]
        with MODEL_CACHE_LOCK:
            for user_id, model in GLOBAL_MODEL_CACHE.items():
                if hasattr(model, "saved_means") and len(model.saved_means) == NUM_FEATURES:
                    self.sources.append(user_id); means.append(model.saved_means.tolist()); stds.append(model.saved_scales.tolist()); versions.append(GLOBAL_TRAIN_COUNTER.get(user_id, 0))
        self.gpu_means = torch.tensor(means, device="cuda", dtype=torch.float32)
        self.gpu_stds = torch.tensor(stds, device="cuda", dtype=torch.float32)
        self.gpu_versions = torch.tensor(versions, device="cuda", dtype=torch.int32)
        source_series = cudf.Series(self.sources); hashes_cupy = source_series.hash_values().values 
        hashes_torch = torch.as_tensor(hashes_cupy, device="cuda").to(torch.int64)
        self.sorted_hashes, sort_order = torch.sort(hashes_torch)
        self.sorted_indices = torch.arange(len(self.sources), device="cuda")[sort_order]
        avg_v = sum(versions) / len(versions) if versions else 0
        logger.info(f"✅ GPU MODELS READY. Profiles: {len(self.sources)} | Max Version: {max(versions) if versions else 0}")
    @property
    def name(self): return "manual-inference"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(MessageMeta); return schema
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        try:
            if MODEL_UPDATED_EVENT.is_set() and (pytime.time() - self._last_reload > self._min_reload_cooldown):
                MODEL_UPDATED_EVENT.clear(); self._build_gpu_lookups(); self._last_reload = pytime.time()
            
            # Process the batch directly from the ControlMessage (AccumulatorStage handled batching)
            big_df = msg.payload().df
            input_tensor = torch.as_tensor(big_df[FEATURE_COLS].values, device='cuda', dtype=torch.float32)
            batch_hashes = torch.as_tensor(big_df["source"].hash_values().values, device="cuda").to(torch.int64)
            search_idx = torch.searchsorted(self.sorted_hashes, batch_hashes).clamp(0, len(self.sorted_hashes) - 1)
            stats_ids = torch.where(self.sorted_hashes[search_idx] == batch_hashes, self.sorted_indices[search_idx], torch.tensor(0, device="cuda"))
            batch_means = self.gpu_means[stats_ids]; batch_stds = self.gpu_stds[stats_ids]
            normalized_input = (input_tensor - batch_means) / batch_stds
            
            reconstruction = torch.zeros_like(normalized_input)
            unique_ids = stats_ids.unique()
            
            # OPTIMIZATION: Take lock once per batch to gather all required models
            # This significantly reduces overhead when a batch contains many unique hosts
            models_to_run = {}
            with MODEL_CACHE_LOCK:
                for sid in unique_ids:
                    sid_item = int(sid.item())
                    if sid_item == 0: continue
                    user_id = self.sources[sid_item]
                    model = GLOBAL_MODEL_CACHE.get(user_id)
                    if model: models_to_run[sid_item] = model
            
            for sid_item, model in models_to_run.items():
                mask = (stats_ids == sid_item)
                with torch.no_grad():
                    try:
                        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                            out = model.model(normalized_input[mask])
                            if isinstance(out, tuple):
                                target_width = normalized_input[mask].shape[1]
                                for o in out:
                                    if hasattr(o, "shape") and len(o.shape) > 1 and o.shape[1] == target_width:
                                        reconstruction[mask] = o
                                        break
                            else:
                                reconstruction[mask] = out
                        elif hasattr(model, "reconstruct"):
                            reconstruction[mask] = model.reconstruct(normalized_input[mask])
                        else:
                            reconstruction[mask] = model(normalized_input[mask])
                    except Exception:
                        reconstruction[mask] = normalized_input[mask]
            
            with torch.no_grad():
                abs_errors = torch.abs(normalized_input - reconstruction)

                # Exclude time features from both scoring and reason attribution.
                weighted_errors = abs_errors * self._feature_weights
                max_err = torch.max(weighted_errors, dim=1).values
                mean_err = torch.mean(weighted_errors, dim=1)

                # BLENDED LOSS (0.7 MAE + 0.3 MAX): Captures both general drift and extreme spikes
                loss_f32 = (0.7 * mean_err + 0.3 * max_err).to(torch.float32)

                # REASON ATTRIBUTION: Use relative error on non-time features only.
                # Time features are masked to -inf so they can never be selected.
                feat_median = torch.median(abs_errors, dim=0).values.unsqueeze(0)
                feat_iqr = (torch.quantile(abs_errors, 0.75, dim=0) - torch.quantile(abs_errors, 0.25, dim=0)).unsqueeze(0).clamp(min=0.01)
                relative_errors = (abs_errors - feat_median) / feat_iqr
                relative_errors[:, 4] = float('-inf')
                relative_errors[:, 5] = float('-inf')
                relative_errors[:, 6] = float('-inf')
                relative_errors[:, 7] = float('-inf')
                _, reason_indices = torch.max(relative_errors, dim=1)
                reason_codes = reason_indices.cpu().numpy()
            
            # DEFINE LEARNING STATE: Host is new (no model) or untrained (version 0).
            # Version >= 1 means a model exists and has been fit at least once — score it.
            batch_v = self.gpu_versions[stats_ids].cpu().numpy()
            is_learning = (stats_ids == 0).cpu().numpy() | (batch_v < 1)
            
            scores_cp = cp.asarray(loss_f32)
            is_learning_cp = cp.asarray(is_learning)
            
            if len(big_df) > 0:
                batch_avg_raw_loss = float(cp.mean(scores_cp))
                PIPELINE_ANOMALY_LOSS.labels(pipeline_type="syslog").observe(batch_avg_raw_loss)
            
            # DIAGNOSTIC LOGGING (Throttled): See why scores are hitting 0 or 10
            if len(big_df) > 0 and self._debug_counter % 50 == 0:
                avg_raw_loss = float(cp.mean(scores_cp))
                max_raw_loss = float(cp.max(scores_cp))
                learning_count = int(cp.sum(is_learning_cp))
                logger.info(f"🔬 SCORING DIAG: Batch={len(big_df)} | RawLoss (Avg={avg_raw_loss:.3f}, Max={max_raw_loss:.3f}) | Learning={learning_count}")
                if not is_learning_cp.all():
                    # Sample a non-learning row
                    idx = int(cp.where(~is_learning_cp)[0][0])
                    sample_raw = float(scores_cp[idx])
                    sample_mean = float(mean_err[idx])
                    sample_max = float(max_err[idx])
                    sample_feat = FEATURE_COLS[int(reason_codes[idx])]
                    logger.info(f"   Sample Row {idx}: Raw={sample_raw:.3f} (MAE={sample_mean:.3f}, Max={sample_max:.3f} on {sample_feat})")
            self._debug_counter += 1
            
            # REFINED SCALING: Smoother curve to prevent the 0.0 or 10.0 binary behavior
            # Map score to 0-10 range with a sigmoid-like curve for better forensic detail
            # RawLoss of 1.0 (1 std floor) -> ~2.0
            # RawLoss of 5.0 (5 std floor) -> ~5.5
            # RawLoss of 20.0 (20 std floor) -> ~9.0
            scaled_scores = cp.where(scores_cp <= 1.5, 
                                     scores_cp * 2.0, # 0.0-1.5 maps to 0.0-3.0
                                     3.0 + cp.log1p(cp.maximum(0, scores_cp - 1.5)) * 2.0) # smoother log growth
            
            scaled_scores = cp.where(is_learning_cp, 0.0, cp.clip(scaled_scores, 0.0, 10.0))
            
            # 6. REASON MAPPING - STAY ON GPU
            reason_map = {0: "Volume Anomaly", 1: "Severity Spike", 2: "Unusual Event", 3: "Security Flag",
                         4: "Time Anomaly (Hour)", 5: "Time Anomaly (Hour)", 6: "Time Anomaly (Day)", 7: "Time Anomaly (Day)",
                         8: "User Anomaly", 9: "Message Length Anomaly", 10: "Process Anomaly"}
            for i in range(11, NUM_FEATURES): reason_map[i] = "Log Content Anomaly"
            
            # Update results directly in cuDF (Zero-Copy)
            big_df["anomaly_score"] = scaled_scores
            big_df["mean_abs_err"] = cp.asarray(mean_err)
            
            # Vectorized reason mapping using cuDF
            reason_series = cudf.Series(reason_codes)
            big_df["anomaly_cause"] = reason_series.map(reason_map).fillna("Unknown")
            
            # Handle "Learning..." state
            is_learning_gdf = cudf.Series(is_learning_cp)
            big_df.loc[is_learning_gdf, "anomaly_cause"] = "Learning..."
            
            new_meta = MessageMeta(big_df); self._gc_anchor.append(new_meta); return new_meta
        except Exception as e: logger.error(f"❌ Inference Error: {e}"); return None
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("threat-pattern")
class ThreatPatternStage(PassThruTypeMixin, SinglePortStage):
    """Maintains a per-host sliding window of recent anomaly causes and scores,
    then matches known attack patterns to produce a 'threat_context' field.
    Lightweight CPU work — no GPU tensors, just dict lookups on the batch."""

    WINDOW_SECONDS = 300  # 5-minute lookback
    MAX_WINDOW_SIZE = 200

    # Each pattern: (name, description, matcher_fn)
    # matcher_fn receives (reasons: Counter, scores: list, window_size: int)
    # and returns True if the pattern matches.
    PATTERNS = []

    def __init__(self, c):
        super().__init__(c)
        # {source: deque of (timestamp, score, reason)}
        self._windows = collections.defaultdict(collections.deque)

    @property
    def name(self): return "threat-pattern"
    def accepted_types(self): return (MessageMeta,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(MessageMeta); return schema

    @staticmethod
    def _define_patterns():
        """Pattern definitions. Each returns (label, description, confidence) or None.
        Confidence is 0.0–1.0 based on how far the signal exceeds the threshold."""
        patterns = []

        def brute_force(reasons, _scores, n):
            sec_count = reasons.get("Security Flag", 0)
            ratio = sec_count / max(n, 1)
            if sec_count >= 5 and ratio > 0.3:
                conf = min(1.0, 0.4 + (sec_count - 5) * 0.06 + (ratio - 0.3) * 1.0)
                return ("Brute Force Attack",
                        f"{sec_count} security events in window — repeated authentication failures indicate credential stuffing or brute force",
                        round(conf, 2))
            return None
        patterns.append(brute_force)

        def lateral_movement(reasons, _scores, _n):
            user_count = reasons.get("User Anomaly", 0)
            sec_count = reasons.get("Security Flag", 0)
            proc_count = reasons.get("Process Anomaly", 0)
            if user_count >= 2 and (sec_count >= 1 or proc_count >= 1):
                signals = min(user_count, 5) + min(sec_count, 3) + min(proc_count, 3)
                conf = min(1.0, 0.3 + signals * 0.07)
                return ("Lateral Movement",
                        "New user accounts with security events or unfamiliar processes — possible attacker pivoting through the network",
                        round(conf, 2))
            return None
        patterns.append(lateral_movement)

        def service_install(reasons, _scores, _n):
            proc_count = reasons.get("Process Anomaly", 0)
            content_count = reasons.get("Log Content Anomaly", 0)
            volume_count = reasons.get("Volume Anomaly", 0)
            if proc_count >= 2 and (content_count >= 3 or volume_count >= 1):
                signals = min(proc_count, 6) + min(content_count, 8) + min(volume_count, 4)
                conf = min(1.0, 0.3 + signals * 0.05)
                return ("New Service Detected",
                        "New processes generating unfamiliar log patterns with volume changes — a new service or application has been installed",
                        round(conf, 2))
            return None
        patterns.append(service_install)

        def data_exfiltration(reasons, scores, _n):
            vol_count = reasons.get("Volume Anomaly", 0)
            high_scores = sum(1 for s in scores if s > 6.0)
            if vol_count >= 3 and high_scores >= 3:
                conf = min(1.0, 0.4 + (vol_count - 3) * 0.08 + (high_scores - 3) * 0.08)
                return ("Possible Data Exfiltration",
                        f"Sustained volume spikes with {high_scores} critical-severity scores — abnormal data transfer pattern",
                        round(conf, 2))
            return None
        patterns.append(data_exfiltration)

        def privilege_escalation(reasons, scores, _n):
            user_count = reasons.get("User Anomaly", 0)
            sec_count = reasons.get("Security Flag", 0)
            high_scores = sum(1 for s in scores if s > 7.0)
            if user_count >= 1 and sec_count >= 2 and high_scores >= 2:
                conf = min(1.0, 0.5 + (sec_count - 2) * 0.1 + (high_scores - 2) * 0.1)
                return ("Privilege Escalation Attempt",
                        "Unknown users triggering high-severity security events — possible privilege escalation or unauthorized access",
                        round(conf, 2))
            return None
        patterns.append(privilege_escalation)

        def crypto_mining(reasons, _scores, _n):
            proc_count = reasons.get("Process Anomaly", 0)
            vol_count = reasons.get("Volume Anomaly", 0)
            if proc_count >= 2 and vol_count >= 2 and not reasons.get("Security Flag", 0):
                conf = min(1.0, 0.3 + (proc_count - 2) * 0.1 + (vol_count - 2) * 0.1)
                return ("Suspicious Process Activity",
                        "New high-volume processes without security flags — possible crypto miner, unauthorized daemon, or resource hijacking",
                        round(conf, 2))
            return None
        patterns.append(crypto_mining)

        def behavioral_drift(reasons, _scores, n):
            distinct = sum(1 for v in reasons.values() if v >= 2)
            if distinct >= 4 and n >= 10:
                conf = min(1.0, 0.4 + (distinct - 4) * 0.15 + (n - 10) * 0.01)
                return ("Major Behavioral Shift",
                        f"{distinct} distinct anomaly categories active — host behavior has fundamentally changed from its learned baseline",
                        round(conf, 2))
            return None
        patterns.append(behavioral_drift)

        return patterns

    def _evaluate_host(self, source, now):
        """Evaluate all patterns for a host and return the best match."""
        window = self._windows.get(source)
        if not window:
            return None, None, 0.0

        # Prune old entries
        cutoff = now - self.WINDOW_SECONDS
        while window and window[0][0] < cutoff:
            window.popleft()

        if len(window) < 3:
            return None, None, 0.0

        reasons = collections.Counter(entry[2] for entry in window)
        scores = [entry[1] for entry in window]
        n = len(window)

        matches = []
        for pattern_fn in self.PATTERNS:
            result = pattern_fn(reasons, scores, n)
            if result:
                matches.append(result)

        if not matches:
            return None, None, 0.0

        # Pick the highest-confidence match
        best = max(matches, key=lambda m: m[2])
        return best

    def _process(self, msg: MessageMeta):
        if msg is None:
            return None
        try:
            with msg.mutable_dataframe() as df:
                now = pytime.time()
                sources = df["source"].to_pandas().values
                scores = df["anomaly_score"].to_pandas().values
                causes = df["anomaly_cause"].to_pandas().values

                # Update windows for hosts with actual anomalies
                for i in range(len(sources)):
                    score = float(scores[i])
                    cause = str(causes[i])
                    if score > 2.0 and cause != "Learning...":
                        src = str(sources[i])
                        window = self._windows[src]
                        window.append((now, score, cause))
                        # Cap window size
                        while len(window) > self.MAX_WINDOW_SIZE:
                            window.popleft()

                # Evaluate patterns for each unique source in this batch
                context_labels = []
                context_descriptions = []
                context_confidences = []
                unique_sources = {}
                for src in set(sources):
                    label, desc, conf = self._evaluate_host(str(src), now)
                    unique_sources[str(src)] = (label or "", desc or "", conf)

                for src in sources:
                    entry = unique_sources.get(str(src), ("", "", 0.0))
                    context_labels.append(entry[0])
                    context_descriptions.append(entry[1])
                    context_confidences.append(entry[2])

                df["threat_context"] = context_labels
                df["threat_description"] = context_descriptions
                df["threat_confidence"] = context_confidences

        except Exception as e:
            logger.error(f"ThreatPattern Error: {e}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

# Initialize patterns once at class definition time
ThreatPatternStage.PATTERNS = ThreatPatternStage._define_patterns()


@register_stage("python-kafka-sink")
class PythonKafkaSink(SinglePortStage):
    def __init__(self, c, bootstrap_servers, output_topic):
        super().__init__(c)
        self._topic = output_topic
        self._producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'linger.ms': 100,
            'compression.type': 'snappy',
            'on_delivery': self._delivery_callback,
        })

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            logger.error(f"❌ Kafka delivery failure: topic={msg.topic()} partition={msg.partition()} error={err}")

    def __del__(self):
        try: self._producer.flush(timeout=30)
        except Exception: pass

    @property
    def name(self): return "python-kafka-sink"
    def accepted_types(self): return (MessageMeta,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(MessageMeta); return schema

    def _process(self, msg: MessageMeta):
        if msg is None: return None
        try:
            with msg.mutable_dataframe() as df:
                pdf = df.to_pandas()
                keys = pdf['source'].values
                json_batch = pdf.to_json(orient='records', lines=True, date_format='iso')
                for key, line in zip(keys, json_batch.splitlines()):
                    if not line:
                        continue
                    # Retry once on BufferError (producer queue full under load burst)
                    for attempt in range(2):
                        try:
                            self._producer.produce(self._topic, value=line.encode('utf-8'), key=str(key).encode('utf-8'))
                            break
                        except BufferError:
                            if attempt == 0:
                                self._producer.poll(1)
                            else:
                                logger.warning(f"⚠️ Kafka producer queue full — dropping message for key={key}")
                self._producer.poll(0)
        except Exception as e:
            logger.error(f"❌ Kafka Sink Error: {e}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("decode-triton-outputs")
class DecodeTritonOutputsStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c, triton_url, model_dir=os.environ.get("INFERENCE_MODELS_DIR", "/app/models/inference")):
        super().__init__(c); self.decode_maps = {}
        try:
            import sklearn.preprocessing
            for key, filename in [("main", "label_encoder.joblib"), ("sec", "is_security_event_label_encoder.joblib"), ("type", "event_type_label_encoder.joblib"), ("prio", "priority_label_encoder.joblib")]:
                path = os.path.join(model_dir, filename)
                if os.path.exists(path):
                    enc = joblib.load(path)
                    if isinstance(enc, sklearn.preprocessing.LabelEncoder): self.decode_maps[key] = {i: label for i, label in enumerate(enc.classes_)}
        except: pass
    @property
    def name(self): return "decode-triton-outputs"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _build_single(self, builder, input_node):
        def decode(msg):
            if msg:
                with msg.payload().mutable_dataframe() as df:
                    decoded_cols = []
                    for col, tgt, key in [("label","main_label","main"),("is_security_event","is_security_event_str","sec"),("event_type","event_type_str","type"),("priority","priority_str","prio")]:
                        if col in df.columns and key in self.decode_maps:
                            df[tgt] = df[col].astype("int32").map(self.decode_maps[key])
                            decoded_cols.append(tgt)
                    if decoded_cols:
                        logger.info(f"🏷️ DECODED TRITON OUTPUTS: {decoded_cols} for {len(df)} rows")
                        if len(df) > 0:
                            sample = df[decoded_cols].iloc[0].to_dict()
                            logger.info(f"🧪 SAMPLE DECODE: {sample}")
            return msg
        node = builder.make_node(self.unique_name, ops.map(decode)); builder.make_edge(input_node, node); return node

@register_stage("enforce-consistency")
class EnforceConsistencyStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c): super().__init__(c)
    @property
    def name(self): return "enforce-consistency"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _build_single(self, builder, input_node):
        def enforce(msg):
            if msg:
                with msg.payload().mutable_dataframe() as df:
                    if "main_label" in df.columns and "is_security_event_str" in df.columns:
                        mask = (df['main_label'].fillna("").str.contains("NON_SECURITY")) & (df['is_security_event_str'] == "True")
                        if mask.any(): 
                            df.loc[mask, 'is_security_event_str'] = "False"
                            if "event_type_str" in df.columns:
                                df.loc[mask, 'event_type_str'] = "Normal Nonsecurity Event"
            return msg
        node = builder.make_node(self.unique_name, ops.map(enforce)); builder.make_edge(input_node, node); return node

class StringTritonInferenceStage(SinglePortStage):
    def __init__(self, c, model_name, server_url, output_mapping):
        super().__init__(c)
        self._model_name = model_name
        self._server_url = server_url
        self.output_mapping = output_mapping
        self._client = None
        self._debug_counter = 0

    @property
    def name(self): return "string-triton-inference"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema):
        schema.output_schema.set_type(ControlMessage)
        return schema
    
    def _ensure_client(self):
        from tritonclient.grpc import InferenceServerClient
        if self._client is None:
            try:
                # Sanitize URL: Remove http:// and trailing slashes for gRPC
                sanitized_url = self._server_url.replace("http://", "").replace("https://", "").rstrip("/")
                self._client = InferenceServerClient(url=sanitized_url, verbose=False)
                logger.info(f"✅ Created Triton client for {sanitized_url}")
            except Exception as e:
                logger.error(f"❌ Failed to create Triton client for {self._server_url}: {e}")
                return False
        try:
            is_live = self._client.is_server_live()
            return is_live
        except Exception:
            self._client = None
            return False

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        # Single mutable_dataframe() context: set triton_classified=False as default,
        # overwrite with True only after a successful Triton response.  Calling
        # mutable_dataframe() twice on the same MessageMeta in the same invocation is
        # fragile in Morpheus 25.x and was the likely cause of the column-size mismatch.
        with msg.payload().mutable_dataframe() as df:
            df["triton_classified"] = False
            if self._ensure_client():
                try:
                    batch_size = len(df)
                    do_log = (self._debug_counter % 100 == 0)
                    self._debug_counter += 1
                    if do_log:
                        logger.info(f"🔵 TRITON CALL: model={self._model_name}, batch_size={batch_size}")
                    # to_pandas() is required for strings as to_arrow() enforces zero_copy_only
                    msg_array = df["message"].to_pandas().values.astype(object).reshape(-1, 1)
                    from tritonclient.grpc import InferInput, InferRequestedOutput
                    inputs = [InferInput("input_message", msg_array.shape, "BYTES")]
                    inputs[0].set_data_from_numpy(msg_array)
                    outputs = [InferRequestedOutput(out.name) for out in self.output_mapping]
                    response = self._client.infer(self._model_name, inputs, outputs=outputs)
                    if do_log:
                        samples = min(3, batch_size)
                        for i in range(samples):
                            msg_peek = str(msg_array[i][0])[:60].replace("\n", " ")
                            results_peek = {out.name: response.as_numpy(out.name)[i] for out in self.output_mapping}
                            logger.info(f"   Sample {i}: '{msg_peek}...' -> {results_peek}")
                    for out in self.output_mapping:
                        result = response.as_numpy(out.name)
                        if result is not None:
                            df[out.name] = result.flatten()
                            df[f"feature_{out.name}"] = df[out.name].astype("float32")
                    df["triton_classified"] = True
                except Exception as e:
                    logger.error(f"❌ Triton inference failed: {e}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

def run_inference_pipeline():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{MODEL_DIR}"
    # Health server must start first so K8s startup probes get responses
    # during the entire warm-up and CUDA init window.
    threading.Thread(target=run_health_server, daemon=True).start()
    pytime.sleep(1)  # give uvicorn a moment to bind before probes can fire
    warm_up_model_cache()
    threading.Thread(target=memory_manager_worker, daemon=True).start()
    threading.Thread(target=model_janitor_worker, daemon=True).start()
    threading.Thread(target=control_worker, daemon=True).start()
    while True:
        try:
            config = Config(); config.execution_mode = ExecutionMode.GPU; config.pipeline_batch_size = kafka_batch_size; config.edge_buffer_size = kafka_edge_buffer; config.mode = "AE"; config.num_threads = 8
            KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
            KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "morpheus_syslog_dfp_v1")
            INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "kafka_input")
            OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "kafka_output")
            TRITON_URL = os.environ.get("TRITON_SERVER_URL", "localhost:8001")

            pipeline = LinearPipeline(config)
            # CONSISTENT GROUP ID: Allows multi-pod scaling across partitions
            pipeline.set_source(KafkaSource(config, bootstrap_servers=KAFKA_BROKER, input_topic=INPUT_TOPIC, group_id=KAFKA_GROUP_ID, auto_offset_reset="latest"))
            
            pipeline.add_stage(DeserializeStage(config))
            
            # ACCUMULATOR AT THE FRONT: Ensures all downstream stages work on large batches
            # This is critical for high-throughput Triton and GPU Preprocessing
            pipeline.add_stage(AccumulatorStage(config))
            
            pipeline.add_stage(GPUPreprocessingStage(config))
            pipeline.add_stage(StringTritonInferenceStage(config, model_name="cyber_log_pipeline", server_url=TRITON_URL, output_mapping=[ColumnInfo(name="label", dtype="int64"), ColumnInfo(name="is_security_event", dtype="int64"), ColumnInfo(name="event_type", dtype="int64"), ColumnInfo(name="priority", dtype="int64")]))
            pipeline.add_stage(DecodeTritonOutputsStage(config, triton_url=TRITON_URL))
            pipeline.add_stage(EnforceConsistencyStage(config))
            pipeline.add_stage(GPUFeatureMappingStage(config))
            
            pipeline.add_stage(AsyncTrainingStage(config))
            pipeline.add_stage(ManualInferenceStage(config))
            pipeline.add_stage(ThreatPatternStage(config))
            pipeline.add_stage(PythonKafkaSink(config, bootstrap_servers=KAFKA_BROKER, output_topic=OUTPUT_TOPIC))
            logger.info("🚀 Morpheus DFP Inference Pipeline starting..."); PIPELINE_HEALTH["status"] = "running"; pipeline.run()
        except Exception as e: logger.critical(f"🛑 FATAL PIPELINE ERROR: {e}"); PIPELINE_HEALTH["status"] = "error"; pytime.sleep(10)
        except KeyboardInterrupt: break

if __name__ == "__main__": run_inference_pipeline()
