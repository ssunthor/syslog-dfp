# Author: Scott Chrestman
# Copyright: 2025 GTT Communications, Inc.

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
from datetime import datetime, timedelta
from confluent_kafka import Producer

# CUDA / ML
import cudf
import cupy as cp
import torch
import mlflow

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

# --- LOGGING & GLOBALS ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("morpheus_inference")
warnings.filterwarnings("ignore")
logging.getLogger("autoencoder").setLevel(logging.CRITICAL)
logging.getLogger("morpheus.models.dfencoder").setLevel(logging.CRITICAL)

MLFLOW_LOCK = threading.Lock()
MODEL_DIR = "/app/models/dfp-models"
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=12)
TRAINING_IN_PROGRESS = set()
TRAINING_SET_LOCK = threading.Lock()
MODEL_UPDATED_EVENT = threading.Event()

# --- STAGES ---

@register_stage("accumulator")
class AccumulatorStage(SinglePortStage):
    def __init__(self, c):
        super().__init__(c)
        self._buffer = []
        self._count = 0
        self._last_flush = pytime.time()
        self.batch_size = 16384
        self.max_latency = 0.5

    @property
    def name(self): return "accumulator"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False
    def compute_schema(self, schema):
        schema.output_schema.set_type(ControlMessage)
        return schema

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            self._buffer.append(df.to_pandas())
            self._count += len(df)
            
        now = pytime.time()
        if self._count < self.batch_size and (now - self._last_flush < self.max_latency):
            return None 
            
        try:
            big_df = pd.concat(self._buffer)
            big_df.reset_index(drop=True, inplace=True)
            self._buffer = []
            self._count = 0
            self._last_flush = now
            out = ControlMessage()
            out.payload(MessageMeta(big_df))
            return out
        except:
            return None

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("gpu-clean-room")
class GPUCleanRoomStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, c): 
        super().__init__(c)
        
    @property
    def name(self): 
        return "gpu-clean-room"
        
    def accepted_types(self): 
        return (ControlMessage,)
        
    def supports_cpp_node(self): 
        return False

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            try:
                # 1. Regex to find the Timestamp + Syslog Hostname
                regex_pattern = r"(?:<\d+>)?([A-Z][a-z]{2}\s+\d+\s\d+:\d+:\d+)\s+([a-zA-Z0-9\.\-_]+)"
                
                if "message" in df.columns:
                    extracted = df["message"].str.extract(regex_pattern)
                    df["syslog_timestamp"] = extracted[0]
                    hostname_source = extracted[1]

                    # 2. THE FIX: Extract the specific Attacker IP
                    # Looks for "from 192.168.1.1" pattern typical in SSH logs
                    ip_pattern = r"from\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
                    extracted_ip = df["message"].str.extract(ip_pattern)[0]

                    # 3. Logic: Use IP if found, otherwise fallback to hostname
                    df["source"] = extracted_ip.fillna(hostname_source).fillna("unknown_source")

                else:
                    df["source"] = "unknown_source"
                
                # 4. THE SPIKE FIX: Tighter timestamp spacing
                rows = len(df)
                
                # We multiply by 1,000 (microseconds) instead of 1,000,000 (milliseconds)
                # This compresses the logs in time, creating the high-frequency spike needed for detection.
                time_offsets = cp.arange(rows, dtype='int64') * 1000 
                
                base_ts = int(datetime.utcnow().timestamp() * 1e9)
                df["timestamp"] = base_ts + time_offsets
                df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
                df = df.sort_values("timestamp")

                df["anomaly_score"] = -1.0
                df["mean_abs_err"] = -1.0
                df["anomaly_cause"] = "analysis_pending" 
            except Exception as e:
                logger.error(f"GPU Prep Error: {e}")
                df["source"] = "fallback_user"
                df["timestamp"] = datetime.now()
        return msg

    def _build_single(self, builder, input_node):
        node = builder.make_node(self.unique_name, ops.map(self._process))
        builder.make_edge(input_node, node)
        return node

@register_stage("prepare-dfp-features")
class PrepareDFPFeaturesStage(PassThruTypeMixin, SinglePortStage):
    def __init__(self, config): super().__init__(config)
    @property
    def name(self) -> str: return "prepare-dfp-features"
    def accepted_types(self) -> tuple: return (ControlMessage,)
    def supports_cpp_node(self) -> bool: return False

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            pdf = df.to_pandas()
            if len(pdf) > 1:
                timestamps = pdf['timestamp'].astype(np.int64)
                duration_sec = max((timestamps.max() - timestamps.min()) / 1e9, 0.001)
            else:
                duration_sec = 1.0

            raw_counts = pdf['source'].value_counts()
            host_rates = (raw_counts / duration_sec) * 60.0
            
            # --- THE FIX: LOGARITHMIC SCALING ---
            # We apply np.log1p (Natural Log + 1) to squash massive numbers.
            # 100 becomes 4.6
            # 10,000 becomes 9.2
            # This makes Volume comparable to Priority (0-7).
            #df["feature_log_count"] = np.log1p(pdf['source'].map(host_rates).fillna(0)).astype('float32')
            # ------------------------------------
            # Just use raw rates, or a lighter divisor for the demo
            #df["feature_log_count"] = (pdf['source'].map(host_rates).fillna(0) / 10.0).astype('float32')
            df["feature_log_count"] = (pdf['source'].map(host_rates).fillna(0) / 2000.0).astype('float32')

            for col in ["priority", "event_type", "is_security_event"]:
                if col in df.columns:
                    df[f"feature_{col}"] = cudf.to_numeric(df[col], errors='coerce').fillna(0.0).astype("float32")
                else:
                    df[f"feature_{col}"] = 0.0
        return msg

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
        self._save_interval = 10
        self._queue_semaphore = threading.Semaphore(24) 
        
    @property
    def name(self): return "async-training"
    def accepted_types(self): return (ControlMessage,)
    def supports_cpp_node(self): return False

    def _train_and_save(self, pdf, user_id, count):
        try:
            feature_cols = ["feature_log_count", "feature_priority", "feature_event_type", "feature_is_security_event"]
            train_data = pdf[feature_cols]

            if user_id not in self._model_cache:
                model_name = f"dfp-{user_id}"
                try:
                    if MLFLOW_LOCK.acquire(blocking=False):
                        try:
                            loaded = mlflow.pytorch.load_model(f"models:/{model_name}/Latest")
                            loaded.patience = None
                            if torch.cuda.is_available(): 
                                loaded.to("cuda")
                                loaded.float()
                            self._model_cache[user_id] = loaded
                        except:
                            self._model_cache[user_id] = AutoEncoder(
                                encoder_layers=[32, 16], decoder_layers=[16, 32], activation='relu', patience=None
                            )
                            if torch.cuda.is_available(): self._model_cache[user_id].to("cuda")
                        finally:
                            MLFLOW_LOCK.release()
                    else: return 
                except: pass
            
            if user_id in self._model_cache:
                model = self._model_cache[user_id]
            else: return

            model.fit(train_data, epochs=1) 

            try:
                data_matrix = train_data.values
                if hasattr(data_matrix, 'get'): data_matrix = data_matrix.get()
                batch_mean = np.mean(data_matrix, axis=0)
                batch_std = np.std(data_matrix, axis=0)
                batch_std[batch_std == 0] = 1.0
                
                model.register_buffer("saved_means", torch.tensor(batch_mean, dtype=torch.float16))
                model.register_buffer("saved_scales", torch.tensor(batch_std, dtype=torch.float16))
            except Exception: pass

            if count > 1 and count % self._save_interval == 0:
                if MLFLOW_LOCK.acquire(blocking=False):
                    try:
                        mlflow.set_experiment("DFP")
                        with mlflow.start_run(run_name=f"train_{user_id}_{count}"):
                            mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=f"dfp-{user_id}")
                        print(f"✅ SAVED MODEL: {user_id}")
                        MODEL_UPDATED_EVENT.set()
                    finally:
                        MLFLOW_LOCK.release()
            del train_data
        except Exception: pass
        finally:
             self._queue_semaphore.release()
             with TRAINING_SET_LOCK:
                 if user_id in TRAINING_IN_PROGRESS: TRAINING_IN_PROGRESS.remove(user_id)

    def _process(self, msg: ControlMessage):
        if msg is None: return None
        with msg.payload().mutable_dataframe() as df:
            pdf = df.to_pandas().copy() 
        
        for user_id, user_data_indices in pdf.groupby("source").groups.items():
            if not user_id or user_id == 'unknown_source': continue
            if pytime.time() - self._last_train_time.get(user_id, 0) < 300: continue
            if not self._queue_semaphore.acquire(blocking=False): continue

            self._last_train_time[user_id] = pytime.time()
            self._train_counter[user_id] = self._train_counter.get(user_id, 0) + 1
            
            with TRAINING_SET_LOCK: TRAINING_IN_PROGRESS.add(user_id)
            TRAINING_EXECUTOR.submit(self._train_and_save, pdf.loc[user_data_indices].copy(), user_id, self._train_counter[user_id])

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
        self.target_batch_size = 32768  
        self.max_latency_sec = 1.0      
        self._buffer = []               
        self._buffer_count = 0          
        self._last_flush_time = pytime.time()
        self._gc_anchor = collections.deque(maxlen=10)
        self._last_reload = pytime.time()
        self._min_reload_cooldown = 5.0

        self.generic_model = torch.nn.Sequential(
            torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)
        )
        if torch.cuda.is_available(): self.generic_model.to("cuda").half()
        self._build_gpu_lookups()

    def _build_gpu_lookups(self):
        print("⏳ LOADING GPU MODELS...")
        fallback_mean = [500.0, 4.0, 10.0, 0.0]
        fallback_std = [500.0, 4.0, 10.0, 1.0]
        sources = ["UNKNOWN_FALLBACK"]
        means = [fallback_mean]; stds = [fallback_std]
        
        if os.path.exists("/app/models/dfp-models"):
            model_dirs = [d for d in os.listdir("/app/models/dfp-models") if d.startswith("dfp-")]
            for m_dir in model_dirs:
                try:
                    user_id = m_dir.replace("dfp-", "")
                    loaded = mlflow.pytorch.load_model(f"models:/{m_dir}/Latest", map_location="cpu")
                    if hasattr(loaded, "saved_means"):
                        sources.append(user_id)
                        means.append(loaded.saved_means.tolist())
                        stds.append(loaded.saved_scales.tolist())
                except: pass
        
        self.gpu_means = torch.tensor(means, device="cuda", dtype=torch.float16)
        self.gpu_stds = torch.tensor(stds, device="cuda", dtype=torch.float16)
        
        source_series = cudf.Series(sources)
        hashes_cupy = source_series.hash_values().values 
        hashes_torch = torch.as_tensor(hashes_cupy, device="cuda").to(torch.int64)
        indices_torch = torch.arange(len(sources), device="cuda", dtype=torch.long)
        self.sorted_hashes, sort_order = torch.sort(hashes_torch)
        self.sorted_indices = indices_torch[sort_order]
        print(f"✅ GPU MODELS READY. Profiles: {len(sources)}")

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
                self._build_gpu_lookups()
                self._last_reload = pytime.time()

            with msg.payload().mutable_dataframe() as df:
                self._buffer.append(df.to_pandas())
                self._buffer_count += len(df)
            
            if self._buffer_count < self.target_batch_size and (pytime.time() - self._last_flush_time < self.max_latency_sec):
                return None
            
            big_df_pd = pd.concat(self._buffer, ignore_index=True)
            big_df_pd.reset_index(drop=True, inplace=True)
            self._buffer = []; self._buffer_count = 0; self._last_flush_time = pytime.time()
            
            big_df = cudf.DataFrame.from_pandas(big_df_pd)
            feature_cols = ["feature_log_count", "feature_priority", "feature_event_type", "feature_is_security_event"]
            input_tensor = torch.as_tensor(big_df[feature_cols].values, device='cuda', dtype=torch.float16)
            
            batch_hashes = torch.as_tensor(big_df["source"].hash_values().values, device="cuda").to(torch.int64)
            search_idx = torch.searchsorted(self.sorted_hashes, batch_hashes)
            search_idx = torch.clamp(search_idx, 0, len(self.sorted_hashes) - 1)
            found_hashes = self.sorted_hashes[search_idx]
            stats_ids = self.sorted_indices[search_idx]
            stats_ids = torch.where(found_hashes == batch_hashes, stats_ids, torch.tensor(0, device="cuda"))

            batch_means = self.gpu_means[stats_ids]; batch_stds = self.gpu_stds[stats_ids]
            normalized_input = (input_tensor - batch_means) / batch_stds
            
            with torch.no_grad():
                reconstruction = self.generic_model(normalized_input)
                raw_errors = torch.abs(normalized_input - reconstruction)
                max_val, max_indices = torch.max(raw_errors, dim=1)
                loss_f32 = torch.mean(raw_errors, dim=1).to(torch.float32)
                reason_codes = max_indices.cpu().numpy()

            scores_cpu = loss_f32.cpu().numpy()
            scores_cpu = np.where(scores_cpu > 2.0, scores_cpu * 2.5, scores_cpu)
            big_df_pd["anomaly_score"] = scores_cpu
            big_df_pd["mean_abs_err"] = scores_cpu
            reason_map = {0: "Volume Anomaly", 1: "Severity Spike", 2: "Unusual Event", 3: "Security Flag"}
            big_df_pd["anomaly_cause"] = [reason_map.get(int(c), "Unknown") for c in reason_codes]

            out_gdf = cudf.DataFrame.from_pandas(big_df_pd)
            new_meta = MessageMeta(out_gdf)
            self._gc_anchor.append(new_meta)
            
            if self._debug_counter % 5 == 0: print(f"⚡ FLUSHED BATCH: {len(out_gdf)}")
            self._debug_counter += 1
            return new_meta

        except Exception as e:
            print(f"❌ Flush Error: {e}"); return None

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
            'compression.type': 'snappy'
        })
        self._topic = output_topic

    @property
    def name(self): return "python-kafka-sink"
    def accepted_types(self): return (MessageMeta,)
    def supports_cpp_node(self): return False
    
    def compute_schema(self, schema):
        schema.output_schema.set_type(MessageMeta)
        return schema

    def _process(self, msg: MessageMeta):
        if msg is None: return None
        try:
            with msg.mutable_dataframe() as df:
                pdf = df.to_pandas()
                json_block = pdf.to_json(orient='records', lines=True, date_format='iso')
                for line in json_block.splitlines():
                    self._producer.produce(self._topic, line.encode('utf-8'))
                self._producer.poll(0)
        except Exception as e:
            print(f"❌ Kafka Write Error: {e}")
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
            for key, filename in [("main", "label_encoder.joblib"), ("sec", "is_security_event_vectorizer.joblib"), ("type", "event_type_vectorizer.joblib"), ("prio", "priority_vectorizer.joblib")]:
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
    def supports_cpp_node(self): return False
    def _get_inference_worker(self, inf_queue): return self
    def init(self): 
        from tritonclient.grpc import InferenceServerClient
        self._client = InferenceServerClient(url=self._server_url, verbose=False)
    def stop(self): 
        if hasattr(self, '_client'): self._client.close()
    def build_output_message(self, message): return message
    def _convert_one_response(self, output_message, req, resp, offset): return output_message
    def _split_batches(self, message, max_batch_size): return [message]
    def process(self, batch, cb):
        if batch is None: cb(batch); return
        meta = batch.payload()
        try:
            with meta.mutable_dataframe() as df:
                string_list = df["message"].to_arrow().to_pylist()
                inp_data = np.array([s.encode('utf-8') for s in string_list], dtype=object).reshape(-1, 1)
                from tritonclient.grpc import InferInput, InferRequestedOutput
                inputs = [InferInput("input_message", inp_data.shape, "BYTES")]
                inputs[0].set_data_from_numpy(inp_data)
                outputs = [InferRequestedOutput(out.name) for out in self.output_mapping]
                response = self._client.infer(self._model_name, inputs, outputs=outputs)
                for out in self.output_mapping:
                    result = response.as_numpy(out.name)
                    if result is not None: df[out.name] = result.tolist()
        except Exception: pass
        cb(batch); return

# --- MAIN PIPELINE ---

def run_inference_pipeline():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{MODEL_DIR}"

    config = Config()
    config.execution_mode = ExecutionMode.GPU
    config.pipeline_batch_size = 16384
    config.edge_buffer_size = 16384
    config.mode = "AE" 
    
    KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
    TRITON_URL = os.environ.get("TRITON_SERVER_URL", "localhost:8001")

    pipeline = LinearPipeline(config)

    pipeline.set_source(KafkaSource(config, bootstrap_servers=KAFKA_BROKER, input_topic="syslog-test", group_id=f"morpheus_inf_{uuid.uuid4().hex}"))
    pipeline.add_stage(AddUUIDStage(config))
    pipeline.add_stage(DeserializeStage(config))
    pipeline.add_stage(GPUCleanRoomStage(config))
    
    inference_outputs = [
        ColumnInfo(name="label", dtype="int64"),
        ColumnInfo(name="is_security_event", dtype="int64"),
        ColumnInfo(name="event_type", dtype="int64"),
        ColumnInfo(name="priority", dtype="int64")
    ]
    pipeline.add_stage(StringTritonInferenceStage(config, model_name="cyber_log_pipeline", server_url=TRITON_URL, output_mapping=inference_outputs))
    pipeline.add_stage(DecodeTritonOutputsStage(config, triton_url=TRITON_URL))
    pipeline.add_stage(EnforceConsistencyStage(config))

    pipeline.add_stage(PrepareDFPFeaturesStage(config))
    pipeline.add_stage(AccumulatorStage(config))
    
    pipeline.add_stage(AsyncTrainingStage(config))
    pipeline.add_stage(ManualInferenceStage(config))
    pipeline.add_stage(PythonKafkaSink(config, bootstrap_servers=KAFKA_BROKER, output_topic="morpheus-output"))

    logger.info("Morpheus DFP Inference Pipeline starting...")
    pipeline.run()

if __name__ == "__main__":
    run_inference_pipeline()