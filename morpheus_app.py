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

# Global Cache to share between stages
GLOBAL_MODEL_CACHE = {}
GLOBAL_TRAIN_COUNTER = {}
MODEL_CACHE_LOCK = threading.Lock()

def warm_up_model_cache():
    if not os.path.exists(MODEL_DIR): return
    logger.info(f"🕵️ WARMING UP: Scanning {MODEL_DIR} for existing models...")
    count = 0
    dirs_to_check = [MODEL_DIR, os.path.join(MODEL_DIR, "models")]
    for base_dir in dirs_to_check:
        if not os.path.exists(base_dir): continue
        try:
            for host_dir_name in os.listdir(base_dir):
                if not host_dir_name.startswith("dfp-"): continue
                user_id = host_dir_name.replace("dfp-", "")
                if user_id in GLOBAL_MODEL_CACHE: continue
                host_path = os.path.join(base_dir, host_dir_name)
                try:
                    versions = [v for v in os.listdir(host_path) if v.startswith("version-")]
                    if not versions: continue
                    latest_v_str = sorted(versions, key=lambda x: int(x.split('-')[1]))[-1]
                    latest_v_num = int(latest_v_str.split('-')[1])
                    model_path = os.path.join(host_path, latest_v_str, "model")
                    if os.path.exists(model_path):
                        with MLFLOW_LOCK:
                            loaded = mlflow.pytorch.load_model(model_path)
                            if torch.cuda.is_available(): loaded.to("cuda").float()
                            GLOBAL_MODEL_CACHE[user_id] = loaded
                            GLOBAL_TRAIN_COUNTER[user_id] = latest_v_num
                            count += 1
                except: pass
        except: pass
    logger.info(f"🚀 WARM UP COMPLETE: Loaded {count} models into memory.")

def memory_manager_worker():
    while True:
        try:
            if psutil.virtual_memory().percent > 60:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
            pytime.sleep(10)
        except: pytime.sleep(30)

# --- STAGES ---

@register_stage("accumulator")
class AccumulatorStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._buffer = []; self._count = 0; self._last_flush = pytime.time()
        self.batch_size = 16384; self.max_latency = 1.0; self.max_buffer_size = 32768 
        self._perf_last_time = pytime.time()
    @property
    def name(self): return "accumulator"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(ControlMessage); return schema
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        PIPELINE_HEALTH["last_activity"] = pytime.time(); PIPELINE_HEALTH["is_running"] = True; PIPELINE_HEALTH["status"] = "processing"
        gdf = msg.payload().df; self._buffer.append(gdf); self._count += len(gdf)
        now = pytime.time()
        if self._count < self.batch_size and (now - self._last_flush < self.max_latency) and self._count < self.max_buffer_size: return None 
        try:
            big_gdf = cudf.concat(self._buffer); self._buffer = []; self._count = 0; self._last_flush = now
            dt = now - self._perf_last_time
            if dt >= 5.0: logger.info(f"📊 ACCUMULATOR: Flush Size: {big_gdf.shape[0]} | EPS: {(big_gdf.shape[0])/dt:.1f}"); self._perf_last_time = now
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
    @property
    def name(self): return "gpu-preprocessing"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                for col in FEATURE_COLS: df[col] = 0.0
                if "message" in df.columns:
                    msg_col = df["message"]
                    extracted_host = msg_col.str.extract(self._host_regex, expand=True)[1]
                    extracted_ip = msg_col.str.extract(self._ip_regex, expand=True)[0]
                    df["source"] = extracted_ip.fillna(extracted_host).fillna("unknown_source")
                    df["feature_msg_len"] = (msg_col.str.len() / 512.0).astype("float32").clip(0.0, 1.0)
                    users = msg_col.str.extract(self._user_regex, expand=True)[0].fillna("none")
                    procs = msg_col.str.extract(self._process_regex, expand=True)[0].fillna("unknown")
                    df["feature_user_hash"] = (users.hash_values() % 1000).astype("float32") / 1000.0
                    df["feature_proc_hash"] = (procs.hash_values() % 1000).astype("float32") / 1000.0
                    for i in range(NUM_LOG_FEATURES): df[f"feature_log_v{i}"] = (msg_col.hash_values(seed=i) % 1000).astype("float32") / 1000.0
                else: df["source"] = "unknown_source"
                now = datetime.utcnow(); hour = now.hour + now.minute / 60.0; day = now.weekday()
                df["feature_hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
                df["feature_hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
                df["feature_day_sin"] = float(np.sin(2 * np.pi * day / 7.0))
                df["feature_day_cos"] = float(np.cos(2 * np.pi * day / 7.0))
                rows = len(df); time_offsets = cp.arange(rows, dtype='int64') * 1000 
                df["timestamp"] = cudf.to_datetime(time_offsets + int(now.timestamp() * 1e9), unit="ns")
            except Exception as e: logger.error(f"Preprocessing Early Error: {e}")
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("gpu-feature-mapping")
class GPUFeatureMappingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c): super().__init__(c)
    @property
    def name(self): return "gpu-feature-mapping"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                rows = len(df); duration_sec = max(float(rows) * 1000.0 / 1e9, 0.001)
                counts = df['source'].value_counts()
                df['feature_log_count'] = (df['source'].map(counts) / duration_sec * 60.0 / 2000.0).astype('float32')
                for col in ["priority", "event_type", "is_security_event"]:
                    df[f"feature_{col}"] = df[col].astype("float32") if col in df.columns else 0.0
                df["anomaly_score"] = -1.0; df["mean_abs_err"] = -1.0; df["anomaly_cause"] = "analysis_pending" 
            except Exception as e: logger.error(f"Preprocessing Late Error: {e}")
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("async-training")
class AsyncTrainingStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c):
        super().__init__(c); self._train_counter = {}; self._last_train_time = {} 
        self._save_interval = 20; self._queue_semaphore = threading.Semaphore(64) 
    @property
    def name(self): return "async-training"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def _train_and_save(self, pdf, user_id, count):
        try:
            train_data = pdf[FEATURE_COLS]
            with MODEL_CACHE_LOCK:
                if user_id not in GLOBAL_MODEL_CACHE:
                    GLOBAL_MODEL_CACHE[user_id] = AutoEncoder(encoder_layers=[32, 16], decoder_layers=[16, 32], activation='relu', patience=None)
                    if torch.cuda.is_available(): GLOBAL_MODEL_CACHE[user_id].to("cuda").float()
            model = GLOBAL_MODEL_CACHE.get(user_id)
            if not model: return
            with TRAINING_LOCK: model.fit(train_data, epochs=1) 
            
            # SYNC: Update global counter so inference engine knows we have progressed
            with MODEL_CACHE_LOCK:
                GLOBAL_TRAIN_COUNTER[user_id] = max(GLOBAL_TRAIN_COUNTER.get(user_id, 0), count)
            
            # SIGNAL: Notify inference engine that metadata (version/means) has changed
            MODEL_UPDATED_EVENT.set()

            data_matrix = train_data.values
            if hasattr(data_matrix, 'get'): data_matrix = data_matrix.get()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.register_buffer("saved_means", torch.tensor(np.mean(data_matrix, axis=0), dtype=torch.float32, device=device))
            model.register_buffer("saved_scales", torch.tensor(np.std(data_matrix, axis=0) + 0.05, dtype=torch.float32, device=device))
            if count == 1 or (count > 0 and count % self._save_interval == 0):
                with MLFLOW_LOCK:
                    host_dir = os.path.join(MODEL_DIR, f"dfp-{user_id}")
                    if count == 1: shutil.rmtree(host_dir, ignore_errors=True)
                    version_dir = os.path.join(host_dir, f"version-{count}"); model_save_path = os.path.join(version_dir, "model")
                    os.makedirs(version_dir, exist_ok=True)
                    mlflow.pytorch.save_model(model, path=model_save_path, pip_requirements=["torch", "mlflow", "pandas"])
                    MODEL_UPDATED_EVENT.set(); logger.info(f"💾 SAVED: Host {user_id} Version {count}")
        except Exception as e: logger.error(f"❌ Training Thread Error for {user_id}: {e}")
        finally:
             self._queue_semaphore.release()
             with TRAINING_SET_LOCK:
                 if user_id in TRAINING_IN_PROGRESS: TRAINING_IN_PROGRESS.remove(user_id)
    def _process(self, msg: ControlMessage):
        if msg is None: return None
        df = msg.payload().df; source_col = "source" if "source" in df.columns else "hostname" if "hostname" in df.columns else "host" if "host" in df.columns else None
        if not source_col: return msg
        counts = df[source_col].value_counts(); active_hosts = counts[counts >= 100].index.to_arrow().to_pylist()
        if not active_hosts: return msg
        pdf = df[df[source_col].isin(active_hosts)].to_pandas()
        for user_id in active_hosts:
            if pytime.time() - self._last_train_time.get(user_id, 0) < 60: continue
            with TRAINING_SET_LOCK:
                if user_id in TRAINING_IN_PROGRESS: continue
            if not self._queue_semaphore.acquire(blocking=False): continue
            with MODEL_CACHE_LOCK: self._train_counter[user_id] = max(self._train_counter.get(user_id, 0), GLOBAL_TRAIN_COUNTER.get(user_id, 0)) + 1
            count = self._train_counter[user_id]; logger.info(f"⚡ TRAINING START: Host {user_id} ({counts[user_id]}) - V{count}")
            self._last_train_time[user_id] = pytime.time()
            with TRAINING_SET_LOCK: 
                TRAINING_IN_PROGRESS.add(user_id)
            user_data = pdf[pdf[source_col] == user_id].copy(); TRAINING_EXECUTOR.submit(self._train_and_save, user_data, user_id, count)
        return msg
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("manual-inference")
class ManualInferenceStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c); self._debug_counter = 0; self._gc_anchor = collections.deque(maxlen=5) 
        self._last_reload = pytime.time(); self._min_reload_cooldown = 120.0 
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
            for sid in unique_ids:
                sid_item = int(sid.item())
                if sid_item == 0: continue
                mask = (stats_ids == sid)
                user_id = self.sources[sid_item]
                with MODEL_CACHE_LOCK:
                    model = GLOBAL_MODEL_CACHE.get(user_id)
                if model:
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
                raw_errors = torch.abs(normalized_input - reconstruction)
                max_val, max_indices = torch.max(raw_errors, dim=1)
                loss_f32 = torch.mean(raw_errors, dim=1).to(torch.float32); reason_codes = max_indices.cpu().numpy()
            scores_cpu = loss_f32.cpu().numpy(); batch_v = self.gpu_versions[stats_ids].cpu().numpy()
            is_learning = (stats_ids == 0).cpu().numpy() | (batch_v < 5); scores_cpu[is_learning] = 0.0
            def scale_score(s):
                if s <= 0: return 0.0
                if s <= 2.0: return s
                return min(2.0 + np.log1p(s - 2.0) * 2.0, 10.0)
            # 5. POST-PROCESSING (Scaling) - GPU VECTORIZED
            scores_cp = cp.asarray(loss_f32)
            is_learning_cp = cp.asarray(is_learning)
            
            # Vectorized scaling: if s > 2.0: 2.0 + log1p(s-2.0)*2.0, else s
            scaled_scores = cp.where(scores_cp <= 2.0, scores_cp, 2.0 + cp.log1p(cp.maximum(0, scores_cp - 2.0)) * 2.0)
            scaled_scores = cp.where(scores_cp <= 0, 0.0, scaled_scores)
            scaled_scores = cp.where(is_learning_cp, 0.0, cp.clip(scaled_scores, 0.0, 10.0))
            
            # 6. REASON MAPPING - STAY ON GPU
            reason_map = {0: "Volume Anomaly", 1: "Severity Spike", 2: "Unusual Event", 3: "Security Flag",
                         4: "Time Anomaly (Hour)", 5: "Time Anomaly (Hour)", 6: "Time Anomaly (Day)", 7: "Time Anomaly (Day)",
                         8: "User Anomaly", 9: "Message Length Anomaly", 10: "Process Anomaly"}
            for i in range(11, NUM_FEATURES): reason_map[i] = "Log Content Anomaly"
            
            # Update results directly in cuDF (Zero-Copy)
            big_df["anomaly_score"] = scaled_scores
            big_df["mean_abs_err"] = scores_cp
            
            # Vectorized reason mapping using cuDF
            reason_series = cudf.Series(reason_codes)
            big_df["anomaly_cause"] = reason_series.map(reason_map).fillna("Unknown")
            
            # Handle "Learning..." state
            is_learning_gdf = cudf.Series(is_learning_cp)
            big_df.loc[is_learning_gdf, "anomaly_cause"] = "Learning..."
            
            new_meta = MessageMeta(big_df); self._gc_anchor.append(new_meta); return new_meta
        except Exception as e: logger.error(f"❌ Inference Flush Error: {e}"); self._buffer = []; self._buffer_count = 0; return None
    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process)); builder.make_edge(input_node, node); return node

@register_stage("python-kafka-sink")
class PythonKafkaSink(SinglePortStage):
    def __init__(self, c, bootstrap_servers, output_topic):
        super().__init__(c); self._producer = Producer({'bootstrap.servers': bootstrap_servers, 'linger.ms': 100, 'compression.type': 'snappy'})
        self._topic = output_topic
    @property
    def name(self): return "python-kafka-sink"
    def accepted_types(self): return (MessageMeta,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema): schema.output_schema.set_type(MessageMeta); return schema
    def _process(self, msg: MessageMeta):
        if msg is None: return None
        try:
            with msg.mutable_dataframe() as df:
                # OPTIMIZATION: Use to_json() with lines=True for faster serialization
                pdf = df.to_pandas()
                keys = pdf['source'].values
                json_batch = pdf.to_json(orient='records', lines=True, date_format='iso')
                
                for key, line in zip(keys, json_batch.splitlines()):
                    if line:
                        # KEYED PRODUCTION: Ensures host-stickiness across partitions
                        self._producer.produce(self._topic, value=line.encode('utf-8'), key=str(key).encode('utf-8'))
                self._producer.poll(0)
        except Exception as e: logger.error(f"❌ Kafka Sink Error: {e}")
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
        if self._ensure_client():
            try:
                meta = msg.payload()
                with meta.mutable_dataframe() as df:
                    batch_size = len(df)
                    
                    # DEBUG: Throttled Logging (Every 100 batches)
                    do_log = (self._debug_counter % 100 == 0)
                    self._debug_counter += 1
                    
                    if do_log:
                        logger.info(f"🔵 TRITON CALL: model={self._model_name}, batch_size={batch_size}")
                    
                    # to_pandas() is required for strings as to_arrow() often enforces zero_copy_only
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
                            # OPTIMIZATION: Assign numpy array directly to cudf (Zero-copy where possible)
                            df[out.name] = result.flatten()
                            # CRITICAL: Map to feature column so ManualInferenceStage can see it!
                            df[f"feature_{out.name}"] = df[out.name].astype("float32")
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
    warm_up_model_cache(); threading.Thread(target=memory_manager_worker, daemon=True).start(); threading.Thread(target=run_health_server, daemon=True).start()
    while True:
        try:
            config = Config(); config.execution_mode = ExecutionMode.GPU; config.pipeline_batch_size = 32768; config.edge_buffer_size = 32768; config.mode = "AE"; config.num_threads = 8
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
            pipeline.add_stage(PythonKafkaSink(config, bootstrap_servers=KAFKA_BROKER, output_topic=OUTPUT_TOPIC))
            logger.info("🚀 Morpheus DFP Inference Pipeline starting..."); PIPELINE_HEALTH["status"] = "running"; pipeline.run()
        except Exception as e: logger.critical(f"🛑 FATAL PIPELINE ERROR: {e}"); PIPELINE_HEALTH["status"] = "error"; pytime.sleep(10)
        except KeyboardInterrupt: break

if __name__ == "__main__": run_inference_pipeline()
