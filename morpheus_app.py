# Author: Scott Chrestman
# Copyright: 2026 GTT Communications, Inc.

import os
import logging
import joblib
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
from confluent_kafka import Producer

# --- CUDA / ML ---
import cudf
import cupy as cp
import torch
import mlflow

# Initialize RMM with a fixed pool to prevent it from growing indefinitely
try:
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=1 * 1024 * 1024 * 1024, # 1GB
        maximum_pool_size=3 * 1024 * 1024 * 1024  # 3GB
    )
    # Tell PyTorch to be less aggressive with its cache
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.3, 0) # Max 30% of GPU 0
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
import uvicorn

app = FastAPI()
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
    
    # 1. Check if it's still starting up (grace period)
    if not is_running:
        if (now - last_act) > 300: # 5 mins startup timeout
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "unhealthy", "reason": "Pipeline failed to start within timeout"}
        return {"status": "starting", "pipeline_status": PIPELINE_HEALTH["status"]}

    # 2. Check if it's "running" but stalled (no data processed for 10 minutes)
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
# OPTIMIZATION: Reduced from 24 to 4 to prevent GPU thrashing
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=4) 
TRAINING_IN_PROGRESS = set()
TRAINING_SET_LOCK = threading.Lock()
MODEL_UPDATED_EVENT = threading.Event()
# ADDED: Global training lock to prevent multiple threads from thrashing the GPU
TRAINING_LOCK = threading.Lock()

# --- DFP FEATURE DEFINITIONS ---
# We are implementing 19 features as described in the system design
NUM_LOG_FEATURES = 8
# Base Behavioral
FEATURE_COLS = ["feature_log_count", "feature_priority", "feature_event_type", "feature_is_security_event"]
# Temporal (Sine/Cosine encoding for Hour and Day)
FEATURE_COLS += ["feature_hour_sin", "feature_hour_cos", "feature_day_sin", "feature_day_cos"]
# Identity & Metadata
FEATURE_COLS += ["feature_user_hash", "feature_proc_hash", "feature_msg_len"]
# Semantic Hashing (8 features)
for i in range(NUM_LOG_FEATURES):
    FEATURE_COLS.append(f"feature_log_v{i}")
NUM_FEATURES = len(FEATURE_COLS)

# Global Cache to share between stages
GLOBAL_MODEL_CACHE = {}
GLOBAL_TRAIN_COUNTER = {}
MODEL_CACHE_LOCK = threading.Lock()

def warm_up_model_cache():
    """Scans the model directory and pre-loads all latest models into memory."""
    if not os.path.exists(MODEL_DIR):
        logger.warning(f"⚠️ WARM UP: Directory {MODEL_DIR} not found.")
        return

    logger.info(f"🕵️ WARMING UP: Scanning {MODEL_DIR} for existing models...")
    count = 0
    
    # Check both root and /models subfolder
    dirs_to_check = [MODEL_DIR, os.path.join(MODEL_DIR, "models")]
    
    for base_dir in dirs_to_check:
        if not os.path.exists(base_dir): continue
        try:
            logger.info(f"📁 Checking: {base_dir} (Contents: {os.listdir(base_dir)[:5]})")
            for host_dir_name in os.listdir(base_dir):
                if not host_dir_name.startswith("dfp-"): continue
                
                user_id = host_dir_name.replace("dfp-", "")
                if user_id in GLOBAL_MODEL_CACHE: continue

                host_path = os.path.join(base_dir, host_dir_name)
                if not os.path.isdir(host_path): continue
                
                try:
                    versions = [v for v in os.listdir(host_path) if v.startswith("version-")]
                    if not versions: continue
                    
                    latest_v_str = sorted(versions, key=lambda x: int(x.split('-')[1]))[-1]
                    latest_v_num = int(latest_v_str.split('-')[1])
                    model_path = os.path.join(host_path, latest_v_str, "model")
                    
                    if os.path.exists(model_path):
                        with MLFLOW_LOCK:
                            loaded = mlflow.pytorch.load_model(model_path)
                            
                            # ROBUST FEATURE CHECK: Determine if this is a 4-feature or 12-feature model
                            input_size = 0
                            try:
                                # 1. Check saved buffers first (most reliable)
                                if hasattr(loaded, "saved_means"):
                                    input_size = len(loaded.saved_means)
                                # 2. Try to find the first Linear layer in any sub-module
                                if input_size == 0:
                                    for module in loaded.modules():
                                        if isinstance(module, torch.nn.Linear):
                                            input_size = module.in_features
                                            break
                            except Exception:
                                pass
                            
                            if input_size != NUM_FEATURES:
                                logger.info(f"⏭️ Skipping {user_id}: Found {input_size} features, need {NUM_FEATURES} (Old Version)")
                                continue

                            if torch.cuda.is_available(): loaded.to("cuda").float()
                            
                            GLOBAL_MODEL_CACHE[user_id] = loaded
                            GLOBAL_TRAIN_COUNTER[user_id] = latest_v_num
                            count += 1
                except Exception as e:
                    logger.debug(f"Failed to pre-load model for {user_id}: {e}")
        except Exception as e:
            logger.error(f"❌ Error scanning directory {base_dir}: {e}")

    logger.info(f"🚀 WARM UP COMPLETE: Loaded {count} models into memory.")

# --- MEMORY MANAGEMENT ---

def memory_manager_worker():
    """Background thread to monitor memory and only trigger cleanup when needed."""
    logger.info("🧠 Memory Manager Worker started...")
    while True:
        try:
            # 1. Check System RAM (psutil)
            mem_info = psutil.virtual_memory()
            ram_usage_pct = mem_info.percent
            
            # 2. Check GPU Memory (torch)
            gpu_usage_pct = 0
            if torch.cuda.is_available():
                # Use a rough estimate of allocated vs total
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                gpu_usage_pct = (r / t) * 100

            # 3. Trigger Cleanup if Threshold Exceeded (60%)
            # Reduced from 75 to 60 to leave space for ONNX BFC Arena
            if ram_usage_pct > 60 or gpu_usage_pct > 60:
                logger.warning(f"⚠️ Memory Threshold Exceeded! (RAM: {ram_usage_pct}%, GPU: {gpu_usage_pct:.1f}%) - Triggering Cleanup...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Free CuPy/RMM pool blocks
                    cp.get_default_memory_pool().free_all_blocks()
                logger.info("✅ Memory Cleanup Complete.")
            
            # Wait 10s between checks (more aggressive monitoring)
            pytime.sleep(10)
        except Exception as e:
            logger.error(f"Memory Manager Error: {e}")
            pytime.sleep(30)

# --- STAGES ---

@register_stage("accumulator")
class AccumulatorStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._buffer = []
        self._count = 0
        self._last_flush = pytime.time()
        self.batch_size = 4096 # Reduced from 16384
        self.max_latency = 1.0
        self.max_buffer_size = 8192 # Hard cap to prevent memory spikes
        self._perf_last_time = pytime.time()
        self._perf_last_count = 0

    @property
    def name(self): return "accumulator"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema):
        schema.output_schema.set_type(ControlMessage)
        return schema

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        
        # Update pipeline health metrics
        PIPELINE_HEALTH["last_activity"] = pytime.time()
        PIPELINE_HEALTH["is_running"] = True
        PIPELINE_HEALTH["status"] = "processing"

        # Keep data on GPU - use msg.payload().df instead of to_pandas
        gdf = msg.payload().df
        self._buffer.append(gdf)
        self._count += len(gdf)
            
        now = pytime.time()
        if self._count < self.batch_size and (now - self._last_flush < self.max_latency) and self._count < self.max_buffer_size:
            return None 
            
        try:
            start_flush = pytime.perf_counter()
            # Fast GPU Concatenation
            big_gdf = cudf.concat(self._buffer)
            self._buffer = []
            self._count = 0
            self._last_flush = now
            
            # Throughput Logging
            dt = now - self._perf_last_time
            if dt >= 5.0:
                eps = (big_gdf.shape[0]) / (pytime.perf_counter() - start_flush) # Local flush speed
                total_eps = (big_gdf.shape[0]) / dt
                logger.info(f"📊 ACCUMULATOR: Flush Size: {big_gdf.shape[0]} | Throughput: {total_eps:.1f} EPS")
                self._perf_last_time = now

            out = ControlMessage()
            out.payload(MessageMeta(big_gdf))
            return out
        except Exception as e:
            logger.error(f"Accumulator Flush Error: {e}")
            self._buffer = []
            self._count = 0
            return None

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("gpu-preprocessing")
class GPUPreprocessingStage(PassThruTypeMixin, SinglePortStage):
    """Early stage to extract source and generate log content hashes."""
    def __init__(self, c): 
        super().__init__(c)
        self._ip_regex = r"from\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        self._host_regex = r"(?:<\d+>)?([A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+([a-zA-Z0-9\.\-_]+)"
        
    @property
    def name(self): return "gpu-preprocessing"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                # 1. EXTRACT METADATA & SOURCE
                if "message" in df.columns:
                    msg_col = df["message"]
                    extracted_host = msg_col.str.extract(self._host_regex, expand=True)[1]
                    extracted_ip = msg_col.str.extract(self._ip_regex, expand=True)[0]
                    df["source"] = extracted_ip.fillna(extracted_host).fillna("unknown_source")
                    
                    df["feature_user_hash"] = (msg_col.str.extract(r"(?:user|for|by|USER|FOR|BY)\s+([a-zA-Z0-9_\-\.]+)", expand=True)[0].fillna("no_user").hash_values() % 1000).astype("float32") / 1000.0
                    df["feature_proc_hash"] = (msg_col.str.extract(r"\s+([a-zA-Z0-9_\-\.]+)(?:\[\d+\])?:", expand=True)[0].fillna("no_proc").hash_values() % 1000).astype("float32") / 1000.0
                    df["feature_msg_len"] = (msg_col.str.len() / 500.0).astype("float32").clip(0.0, 10.0)

                    for i in range(NUM_LOG_FEATURES):
                        df[f"feature_log_v{i}"] = (msg_col.hash_values(seed=i) % 1000).astype("float32") / 1000.0
                else:
                    df["source"] = "unknown_source"
                    df["feature_user_hash"] = 0.0; df["feature_proc_hash"] = 0.0; df["feature_msg_len"] = 0.0
                    for i in range(NUM_LOG_FEATURES):
                        df[f"feature_log_v{i}"] = 0.0

                # 2. TEMPORAL FEATURES
                now = datetime.utcnow()
                hour = now.hour + now.minute / 60.0
                day = now.weekday()
                df["feature_hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
                df["feature_hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
                df["feature_day_sin"] = float(np.sin(2 * np.pi * day / 7.0))
                df["feature_day_cos"] = float(np.cos(2 * np.pi * day / 7.0))

                # 3. COMPRESSED TIMESTAMPS
                rows = len(df)
                time_offsets = cp.arange(rows, dtype='int64') * 1000 
                base_ts_int = int(now.timestamp() * 1e9)
                df["timestamp"] = cudf.to_datetime(time_offsets + base_ts_int, unit="ns")
            except Exception as e:
                import traceback
                logger.error(f"Preprocessing Early Error: {e}\n{traceback.format_exc()}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("gpu-feature-mapping")
class GPUFeatureMappingStage(PassThruTypeMixin, SinglePortStage):
    """Late stage to map Triton outputs and calculate volume rates."""
    def __init__(self, c): super().__init__(c)
    @property
    def name(self): return "gpu-feature-mapping"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                rows = len(df)
                # Volume features depend on 'source' from Early stage
                duration_sec = max(float(rows) * 1000.0 / 1e9, 0.001)
                counts = df['source'].value_counts()
                df['feature_log_count'] = (df['source'].map(counts) / duration_sec * 60.0 / 2000.0).astype('float32')

                # Triton features depend on Triton inference
                for col in ["priority", "event_type", "is_security_event"]:
                    df[f"feature_{col}"] = df[col].astype("float32") if col in df.columns else 0.0

                df["anomaly_score"] = -1.0
                df["mean_abs_err"] = -1.0
                df["anomaly_cause"] = "analysis_pending" 
            except Exception as e:
                import traceback
                logger.error(f"Preprocessing Late Error: {e}\n{traceback.format_exc()}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node


@register_stage("async-training")
class AsyncTrainingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._model_cache = {} 
        self._train_counter = {}
        self._last_train_time = {} 
        self._save_interval = 5
        self._queue_semaphore = threading.Semaphore(64) 
        self._total_msgs_seen = 0
        
    @property
    def name(self): return "async-training"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False

    def _train_and_save(self, pdf, user_id, count):
        try:
            train_data = pdf[FEATURE_COLS]

            with MODEL_CACHE_LOCK:
                if user_id not in GLOBAL_MODEL_CACHE:
                    model_name = f"dfp-{user_id}"
                    try:
                        host_dir = os.path.join(MODEL_DIR, model_name)
                        if not os.path.exists(host_dir): host_dir = os.path.join(MODEL_DIR, "models", model_name)
                        
                        if os.path.exists(host_dir):
                            versions = [v for v in os.listdir(host_dir) if v.startswith("version-")]
                            if versions:
                                latest_v = sorted(versions, key=lambda x: int(x.split('-')[1]))[-1]
                                model_path = os.path.join(host_dir, latest_v, "model")
                                with MLFLOW_LOCK:
                                    loaded = mlflow.pytorch.load_model(model_path)
                                    
                                    # ROBUST FEATURE CHECK
                                    input_size = 0
                                    try:
                                        if hasattr(loaded, "saved_means"):
                                            input_size = len(loaded.saved_means)
                                        if input_size == 0:
                                            for module in loaded.modules():
                                                if isinstance(module, torch.nn.Linear):
                                                    input_size = module.in_features
                                                    break
                                    except Exception: pass

                                    if input_size != NUM_FEATURES:
                                        logger.info(f"🔄 Migrating {user_id}: Old model has {input_size} features, need {NUM_FEATURES}. Resetting.")
                                        loaded = None

                                    if loaded:
                                        if torch.cuda.is_available(): loaded.to("cuda").float()
                                        GLOBAL_MODEL_CACHE[user_id] = loaded
                                        latest_v_num = int(latest_v.split('-')[1])
                                        GLOBAL_TRAIN_COUNTER[user_id] = max(GLOBAL_TRAIN_COUNTER.get(user_id, 0), latest_v_num)
                    except Exception as e:
                        logger.debug(f"Could not load existing model for {user_id}: {e}")
                    
                    if user_id not in GLOBAL_MODEL_CACHE:
                        GLOBAL_MODEL_CACHE[user_id] = AutoEncoder(
                            encoder_layers=[32, 16], decoder_layers=[16, 32], activation='relu', patience=None
                        )
                        if torch.cuda.is_available(): GLOBAL_MODEL_CACHE[user_id].to("cuda").float()
            
            model = GLOBAL_MODEL_CACHE.get(user_id)
            if not model: return

            # LOCK: Only one model fit can happen on the GPU at a time to prevent thrashing
            with TRAINING_LOCK:
                logger.info(f"🧠 TRAINING: Host {user_id} starting model.fit() on {len(train_data)} logs...")
                model.fit(train_data, epochs=1) 
                logger.info(f"🧠 TRAINING COMPLETE: Host {user_id} fit finished.")
            
            data_matrix = train_data.values
            if hasattr(data_matrix, 'get'): data_matrix = data_matrix.get()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.register_buffer("saved_means", torch.tensor(np.mean(data_matrix, axis=0), dtype=torch.float32, device=device))
            model.register_buffer("saved_scales", torch.tensor(np.std(data_matrix, axis=0) + 1e-4, dtype=torch.float32, device=device))
            
            if count == 1 or (count > 1 and count % self._save_interval == 0):
                logger.info(f"💾 SAVING: Host {user_id} Version {count} to MLflow...")
                with MLFLOW_LOCK:
                    host_dir = os.path.join(MODEL_DIR, f"dfp-{user_id}")
                    if count == 1:
                        logger.info(f"🧹 CLEARING: Host {user_id} starting from version 1. Wiping host directory.")
                        if os.path.exists(host_dir):
                            shutil.rmtree(host_dir, ignore_errors=True)

                    version_dir = os.path.join(host_dir, f"version-{count}")
                    model_save_path = os.path.join(version_dir, "model")
                    
                    try:
                        if os.path.exists(version_dir):
                            shutil.rmtree(version_dir)
                    except Exception as e:
                        logger.warning(f"Could not clear {version_dir}: {e}. Retrying model path only.")
                        if os.path.exists(model_save_path):
                            shutil.rmtree(model_save_path, ignore_errors=True)

                    os.makedirs(version_dir, exist_ok=True)
                    
                    # Log with requirements to be safe
                    mlflow.pytorch.save_model(model, path=model_save_path, pip_requirements=["torch", "mlflow", "pandas"])
                    
                    with MODEL_CACHE_LOCK:
                        GLOBAL_TRAIN_COUNTER[user_id] = max(GLOBAL_TRAIN_COUNTER.get(user_id, 0), count)
                    
                    logger.info(f"💾 SAVED: Host {user_id} Version {count} to disk at {model_save_path}")
                    MODEL_UPDATED_EVENT.set()
            
            del train_data
            del pdf
        except Exception as e:
            logger.error(f"❌ Training Thread Error for {user_id}: {e}")
        finally:
             self._queue_semaphore.release()
             with TRAINING_SET_LOCK:
                 if user_id in TRAINING_IN_PROGRESS: TRAINING_IN_PROGRESS.remove(user_id)

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        self._total_msgs_seen += 1
        
        # OPTIMIZATION: Use cudf for grouping/filtering, only convert subset to pandas
        df = msg.payload().df
        source_col = "source" if "source" in df.columns else "hostname" if "hostname" in df.columns else "host" if "host" in df.columns else None
        if not source_col: return msg

        # 1. Only train on hosts that have a significant amount of data in this batch
        counts = df[source_col].value_counts()
        active_hosts = counts[counts >= 50].index.to_arrow().to_pylist()
        
        if not active_hosts: return msg

        # 2. Extract only the data we need for these hosts
        pdf = df[df[source_col].isin(active_hosts)].to_pandas()

        for user_id in active_hosts:
            last_train = self._last_train_time.get(user_id, 0)
            if pytime.time() - last_train < 60: continue
            
            # CHECK: If already training this host, skip to avoid queue buildup
            with TRAINING_SET_LOCK:
                if user_id in TRAINING_IN_PROGRESS: continue

            if not self._queue_semaphore.acquire(blocking=False): continue

            # SYNC: Initialize or update local counter from global registry to handle restarts
            with MODEL_CACHE_LOCK:
                current_v = max(self._train_counter.get(user_id, 0), GLOBAL_TRAIN_COUNTER.get(user_id, 0))
                self._train_counter[user_id] = current_v + 1

            count = self._train_counter[user_id]
            logger.info(f"⚡ TRAINING START: Host {user_id} (Logs: {counts[user_id]}) - Next Version: {count}")
            self._last_train_time[user_id] = pytime.time()
            
            with TRAINING_SET_LOCK: TRAINING_IN_PROGRESS.add(user_id)
            user_data = pdf[pdf[source_col] == user_id].copy()
            TRAINING_EXECUTOR.submit(self._train_and_save, user_data, user_id, count)

        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("manual-inference")
class ManualInferenceStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._debug_counter = 0
        self.target_batch_size = 4096   
        self.max_latency_sec = 1.0      
        self._buffer = []               
        self._buffer_count = 0          
        self._last_flush_time = pytime.time()
        self._gc_anchor = collections.deque(maxlen=5) 
        self.max_buffer_size = 8192 # Added hard cap
        self._last_reload = pytime.time()
        self._min_reload_cooldown = 120.0 # Increased from 5s to 2m

        # DFP AUTOENCODER: Updated architecture to support log hashing features
        self.generic_model = torch.nn.Sequential(
            torch.nn.Linear(NUM_FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, NUM_FEATURES)
        )
        if torch.cuda.is_available(): self.generic_model.to("cuda").float() # Fixed: Use float32
        self._build_gpu_lookups()

    def _build_gpu_lookups(self):
        logger.info(f"⏳ RELOAD: Syncing GPU from Global Cache ({len(GLOBAL_MODEL_CACHE)} profiles)...")
        fallback_mean = [0.5] * NUM_FEATURES
        fallback_std = [0.1] * NUM_FEATURES
        sources = ["UNKNOWN_FALLBACK"]
        means = [fallback_mean]; stds = [fallback_std]
        
        with MODEL_CACHE_LOCK:
            for user_id, model in GLOBAL_MODEL_CACHE.items():
                if hasattr(model, "saved_means"):
                    # SAFETY: Final check to ensure we don't crash the GPU tensor creation
                    if len(model.saved_means) == NUM_FEATURES:
                        sources.append(user_id)
                        means.append(model.saved_means.tolist())
                        stds.append(model.saved_scales.tolist())
                    else:
                        logger.debug(f"Skipping {user_id} in GPU build: Mean size {len(model.saved_means)} != {NUM_FEATURES}")
        
        logger.info(f"📊 RELOAD COMPLETE. Successfully mapped {len(sources)-1} hosts to GPU models.")
        
        self.gpu_means = torch.tensor(means, device="cuda", dtype=torch.float32)
        self.gpu_stds = torch.tensor(stds, device="cuda", dtype=torch.float32)
        
        source_series = cudf.Series(sources)
        hashes_cupy = source_series.hash_values().values 
        hashes_torch = torch.as_tensor(hashes_cupy, device="cuda").to(torch.int64)
        indices_torch = torch.arange(len(sources), device="cuda", dtype=torch.long)
        self.sorted_hashes, sort_order = torch.sort(hashes_torch)
        self.sorted_indices = indices_torch[sort_order]
        self._debug_registry_samples = sources[1:4] 
        logger.info(f"✅ GPU MODELS READY. Profiles: {len(sources)} | Sample: {sources[:5]}")

    @property
    def name(self): return "manual-inference"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema):
        schema.output_schema.set_type(MessageMeta)
        return schema
    
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        try:
            if MODEL_UPDATED_EVENT.is_set() and (pytime.time() - self._last_reload > self._min_reload_cooldown):
                print(f"🚀 SIGNAL RECEIVED: Reloading GPU Models...")
                MODEL_UPDATED_EVENT.clear()
                try:
                    self._build_gpu_lookups()
                except Exception as e:
                    logger.error(f"❌ Failed to reload GPU models: {e}. Keeping old models.")
                self._last_reload = pytime.time()

            with msg.payload().mutable_dataframe() as df:
                self._buffer.append(df.copy()) # Copy to ensure we don't hold references to message objects
                self._buffer_count += len(df)
            
            now = pytime.time()
            # If buffer getting too big, force a flush regardless of latency
            if self._buffer_count >= self.target_batch_size or (now - self._last_flush_time >= self.max_latency_sec) or self._buffer_count > 100000:
                pass # Proceed to flush
            else:
                return None
            
            # --- START FLUSH TIMING ---
            t_start = pytime.perf_counter()
            
            # 1. FAST GPU CONCAT
            big_df = cudf.concat(self._buffer)
            self._buffer = []; self._buffer_count = 0; self._last_flush_time = now
            t_concat = pytime.perf_counter()
            
            # 2. FEATURE EXTRACTION
            input_tensor = torch.as_tensor(big_df[FEATURE_COLS].values, device='cuda', dtype=torch.float32)
            t_features = pytime.perf_counter()
            
            # 3. HASH LOOKUP
            batch_hashes = torch.as_tensor(big_df["source"].hash_values().values, device="cuda").to(torch.int64)
            search_idx = torch.searchsorted(self.sorted_hashes, batch_hashes)
            search_idx = torch.clamp(search_idx, 0, len(self.sorted_hashes) - 1)
            found_hashes = self.sorted_hashes[search_idx]
            stats_ids = self.sorted_indices[search_idx]
            stats_ids = torch.where(found_hashes == batch_hashes, stats_ids, torch.tensor(0, device="cuda"))
            t_hashes = pytime.perf_counter()

            # DEBUG: Why is lookup 0?
            if self._debug_counter % 10 == 0:
                known_count = (stats_ids > 0).sum().item()
                live_samples = big_df["source"].head(3).to_arrow().to_pylist()
                logger.info(f"🔍 LOOKUP: {known_count}/{len(big_df)} matched. Samples: Live={live_samples} | Registry={self._debug_registry_samples}")

            # 4. NORMALIZATION & INFERENCE
            batch_means = self.gpu_means[stats_ids]; batch_stds = self.gpu_stds[stats_ids]
            normalized_input = (input_tensor - batch_means) / batch_stds
            
            with torch.no_grad():
                reconstruction = self.generic_model(normalized_input)
                raw_errors = torch.abs(normalized_input - reconstruction)
                max_val, max_indices = torch.max(raw_errors, dim=1)
                loss_f32 = torch.mean(raw_errors, dim=1).to(torch.float32)
                reason_codes = max_indices.cpu().numpy()
            t_inference = pytime.perf_counter()
            
            # 5. POST-PROCESSING (Scaling)
            scores_cpu = loss_f32.cpu().numpy()
            is_new_host = (stats_ids == 0).cpu().numpy()
            num_learning = int(is_new_host.sum())
            scores_cpu[is_new_host] = 0.0
            
            def scale_score(s):
                if s <= 0: return 0.0
                if s <= 2.0: return s
                return min(2.0 + np.log1p(s - 2.0) * 2.0, 10.0)

            scores_cpu = np.vectorize(scale_score)(scores_cpu)
            max_batch_score = float(np.max(scores_cpu))
            
            # Update results
            pdf_results = big_df.to_pandas() # Pull back once for cause mapping
            pdf_results["anomaly_score"] = scores_cpu
            pdf_results["mean_abs_err"] = loss_f32.cpu().numpy()
            
            # REASON MAP: Maps the feature index with the highest error to a human-readable cause
            reason_map = {
                0: "Volume Anomaly", 
                1: "Severity Spike", 
                2: "Unusual Event", 
                3: "Security Flag",
                4: "Time Anomaly (Hour)", 
                5: "Time Anomaly (Hour)", 
                6: "Time Anomaly (Day)", 
                7: "Time Anomaly (Day)",
                8: "User Anomaly", 
                9: "Process Anomaly", 
                10: "Message Length Anomaly"
            }
            # Any of the 8 hashing features (indices 11-18) point to log content anomalies
            for i in range(11, 11 + NUM_LOG_FEATURES):
                reason_map[i] = "Log Content Anomaly"

            pdf_results["anomaly_cause"] = [
                "Learning..." if is_new_host[i] else reason_map.get(int(reason_codes[i]), "Unknown")
                for i in range(len(is_new_host))
            ]

            new_meta = MessageMeta(cudf.DataFrame.from_pandas(pdf_results))
            self._gc_anchor.append(new_meta)
            
            t_end = pytime.perf_counter()
            total_ms = (t_end - t_start) * 1000
            
            # DETAILED PERFORMANCE LOGGING
            if self._debug_counter % 5 == 0:
                logger.info(f"📊 INFERENCE: Batch: {len(big_df)} | Max Score: {max_batch_score:.2f} | Learning: {num_learning}/{len(big_df)} hosts")
                logger.info(f"   Breakdown: Concat: {(t_concat-t_start)*1000:.1f}ms | Hash: {(t_hashes-t_features)*1000:.1f}ms | Model: {(t_inference-t_hashes)*1000:.1f}ms")
            
            self._debug_counter += 1
            return new_meta

        except Exception as e:
            logger.error(f"❌ Inference Flush Error: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL RECOVERY: Clear buffer so we don't loop on bad data
            self._buffer = []
            self._buffer_count = 0
            return None

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node); return node

@register_stage("python-kafka-sink")
class PythonKafkaSink(SinglePortStage):
    def __init__(self, c, bootstrap_servers, output_topic):
        super().__init__(c)
        self._producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'queue.buffering.max.messages': 100000,
            'queue.buffering.max.kbytes': 1048576,
            'batch.num.messages': 10000,
            'linger.ms': 100,
            'compression.type': 'snappy',
            'message.max.bytes': 10000000, # 10MB
            'socket.keepalive.enable': True
        })
        self._topic = output_topic
        self._count = 0

    @property
    def name(self): return "python-kafka-sink"
    def accepted_types(self): return (MessageMeta,)
    def supports_cpp_node(self): return False
    
    def compute_schema(self, schema):
        schema.output_schema.set_type(MessageMeta)
        return schema

    def _delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            if self._count % 1000 == 0:
                logger.error(f"❌ Kafka Message delivery failed: {err}")
        self._count += 1

    def _process(self, msg: MessageMeta):
        if msg is None: return None
        try:
            with msg.mutable_dataframe() as df:
                # OPTIMIZATION: Convert to list of dicts once, avoid line-by-line JSON overhead
                records = df.to_pandas().to_dict(orient='records')
                
                import json
                for record in records:
                    # Use standard json.dumps which is faster for single small records than pandas
                    payload = json.dumps(record, default=str).encode('utf-8')
                    try:
                        self._producer.produce(self._topic, payload, callback=self._delivery_report)
                    except BufferError:
                        self._producer.poll(0.1)
                        self._producer.produce(self._topic, payload, callback=self._delivery_report)
                
                # Poll once per batch, not once per record
                self._producer.poll(0)
                
        except Exception as e:
            logger.error(f"❌ Kafka Sink Process Error: {e}")
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node); return node

# --- HELPERS ---

class AddUUIDStage(SinglePortStage):
    def __init__(self, c): super().__init__(c)
    @property
    def name(self): return "add-uuid"
    def accepted_types(self): return (MessageMeta, )
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(MessageMeta); return schema
    def _process(self, msg): 
        msg.set_data("uuid", [str(uuid.uuid4()) for _ in range(msg.count)])
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("decode-triton-outputs")
class DecodeTritonOutputsStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c, triton_url, model_dir="models"):
        super().__init__(c); self.decode_maps = {}
        try:
            import sklearn.preprocessing
            for key, filename in [("main", "label_encoder.joblib"), ("sec", "is_security_event_label_encoder.joblib"), ("type", "event_type_label_encoder.joblib"), ("prio", "priority_label_encoder.joblib")]:
                path = os.path.join(model_dir, filename)
                if os.path.exists(path):
                    enc = joblib.load(path)
                    if isinstance(enc, sklearn.preprocessing.LabelEncoder):
                        self.decode_maps[key] = {i: label for i, label in enumerate(enc.classes_)}
        except: pass
    @property
    def name(self): return "decode-triton-outputs"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _build_single(self, builder, input_node):
        def decode(msg):
            if msg:
                with msg.payload().mutable_dataframe() as df:
                    df["main_label"]="unknown"; df["is_security_event_str"]="False"; df["event_type_str"]="unknown"; df["priority_str"]="unknown"
                    for col, tgt, key in [("label","main_label","main"),("is_security_event","is_security_event_str","sec"),("event_type","event_type_str","type"),("priority","priority_str","prio")]:
                        if col in df.columns and key in self.decode_maps: df[tgt] = df[col].astype("int32").map(self.decode_maps[key])
            return msg
        node = builder.make_node(self.unique_name, ops.map(decode))
        builder.make_edge(input_node, node)
        return node

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
                    if "main_label" in df.columns:
                        mask = (df['main_label'].fillna("").str.contains("NON_SECURITY")) & (df['is_security_event_str'] == "True")
                        if mask.any(): 
                            df.loc[mask, 'is_security_event_str'] = "False"
                            df.loc[mask, 'event_type_str'] = "Normal Nonsecurity Event"
            return msg
        node = builder.make_node(self.unique_name, ops.map(enforce))
        builder.make_edge(input_node, node)
        return node

class StringTritonInferenceStage(TritonInferenceStage):
    def __init__(self, c, model_name, server_url, output_mapping):
        super().__init__(c, model_name=model_name, server_url=server_url, force_convert_inputs=True)
        self.output_mapping = output_mapping
        self._client = None

    def supports_cpp_node(self): return False
    def _get_inference_worker(self, inf_queue): return self
    
    def _ensure_client(self):
        from tritonclient.grpc import InferenceServerClient
        if self._client is None:
            try:
                self._client = InferenceServerClient(url=self._server_url, verbose=False)
            except Exception as e:
                logger.error(f"❌ Failed to create Triton client: {e}")
                return False
        
        try:
            if not self._client.is_server_live():
                logger.warning("⚠️ Triton server is not live. Attempting to reconnect...")
                self._client = InferenceServerClient(url=self._server_url, verbose=False)
                return self._client.is_server_live()
            return True
        except Exception:
            self._client = None
            return False

    def init(self): 
        self._ensure_client()

    def stop(self): 
        if self._client: 
            self._client.close()
            self._client = None

    def build_output_message(self, message): return message
    def _convert_one_response(self, output_message, req, resp, offset): return output_message
    def _split_batches(self, message, max_batch_size): return [message]

    def process(self, batch, cb):
        if batch is None: cb(batch); return
        meta = batch.payload()
        
        success = False
        if self._ensure_client():
            try:
                with meta.mutable_dataframe() as df:
                    msg_array = df["message"].to_arrow().to_numpy().astype(object).reshape(-1, 1)
                    
                    from tritonclient.grpc import InferInput, InferRequestedOutput
                    inputs = [InferInput("input_message", msg_array.shape, "BYTES")]
                    inputs[0].set_data_from_numpy(msg_array)
                    outputs = [InferRequestedOutput(out.name) for out in self.output_mapping]
                    
                    # Retry once on failure
                    for attempt in range(2):
                        try:
                            response = self._client.infer(self._model_name, inputs, outputs=outputs)
                            for out in self.output_mapping:
                                result = response.as_numpy(out.name)
                                if result is not None: df[out.name] = result.tolist()
                            success = True
                            break
                        except Exception as e:
                            if attempt == 0:
                                logger.warning(f"⚠️ Triton inference attempt 1 failed: {e}. Retrying...")
                                self._ensure_client()
                            else:
                                logger.error(f"❌ Triton inference failed after retries: {e}")
            except Exception as e:
                logger.error(f"❌ Triton Stage Error: {e}")

        # Fallback: If Triton failed, ensure columns exist with default values
        if not success:
            try:
                with meta.mutable_dataframe() as df:
                    for out in self.output_mapping:
                        if out.name not in df.columns:
                            df[out.name] = 0 # Default 'unknown' or 'normal' index
            except Exception: pass

        cb(batch); return

# --- MAIN PIPELINE ---

def run_inference_pipeline():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{MODEL_DIR}"

    # Pre-load existing models into memory
    warm_up_model_cache()

    # Start Memory Manager in background
    mem_thread = threading.Thread(target=memory_manager_worker, daemon=True)
    mem_thread.start()

    # Start Health Server in background
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    while True:
        try:
            config = Config()
            config.execution_mode = ExecutionMode.GPU
            config.pipeline_batch_size = 16384 # Reduced from 16384
            config.edge_buffer_size = 16384 # Reduced from 16384
            config.mode = "AE" 
            config.num_threads = 8
            
            KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
            INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "kafka_input")
            OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "kafka_output")
            TRITON_URL = os.environ.get("TRITON_SERVER_URL", "localhost:8001")

            pipeline = LinearPipeline(config)

            pipeline.set_source(KafkaSource(config, bootstrap_servers=KAFKA_BROKER, input_topic=INPUT_TOPIC, group_id=f"morpheus_inf_{uuid.uuid4().hex}"))
            pipeline.add_stage(DeserializeStage(config))
            pipeline.add_stage(GPUPreprocessingStage(config)) # 1. Early Source/Hash Extraction
            
            inference_outputs = [
                ColumnInfo(name="label", dtype="int64"),
                ColumnInfo(name="is_security_event", dtype="int64"),
                ColumnInfo(name="event_type", dtype="int64"),
                ColumnInfo(name="priority", dtype="int64")
            ]
            pipeline.add_stage(StringTritonInferenceStage(config, model_name="cyber_log_pipeline", server_url=TRITON_URL, output_mapping=inference_outputs))
            pipeline.add_stage(DecodeTritonOutputsStage(config, triton_url=TRITON_URL))
            pipeline.add_stage(EnforceConsistencyStage(config))
            pipeline.add_stage(GPUFeatureMappingStage(config)) # 2. Late Feature Mapping (Volume + Triton)
            
            pipeline.add_stage(AsyncTrainingStage(config))
            pipeline.add_stage(AccumulatorStage(config))
            
            pipeline.add_stage(ManualInferenceStage(config))
            pipeline.add_stage(PythonKafkaSink(config, bootstrap_servers=KAFKA_BROKER, output_topic=OUTPUT_TOPIC))

            logger.info("🚀 Morpheus DFP Inference Pipeline starting...")
            PIPELINE_HEALTH["status"] = "running"
            pipeline.run()
        except Exception as e:
            logger.critical(f"🛑 FATAL PIPELINE ERROR: {e}")
            PIPELINE_HEALTH["status"] = "error"
            logger.info("⏳ Restarting pipeline in 10 seconds...")
            pytime.sleep(10)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
            break

if __name__ == "__main__":
    run_inference_pipeline()
