# Syslog DFP (Digital Fingerprinting) Pipeline

## Overview
The **Syslog DFP Pipeline** is a high-performance, real-time log analysis and anomaly detection system built on **NVIDIA Morpheus**. It implements the **Digital Fingerprinting (DFP)** pattern to learn the "normal" behavioral profile of every host in the network and detect deviations that signify security threats or system failures.

The application is designed for **extreme throughput** (thousands of events per second) and follows a **GPU-first architecture**.

---

## ⚡ Core Mandates
- **Performance is Paramount:** Every code change MUST be evaluated for its impact on throughput. Avoid "Python Taxes" such as per-row loops, string encoding in Python, or unnecessary GPU-to-CPU transfers.
- **GPU-Centric Processing:** All heavy lifting—including regex extraction, feature engineering, hashing, and model inference—MUST occur on the GPU using `cudf`, `cupy`, or `torch`.
- **Zero-Copy Philosophy:** Minimize data movement. Keep data in GPU memory (VRAM) as long as possible.

---

## 🛠 Technologies & Infrastructure
- **NVIDIA Morpheus:** The core pipeline engine for GPU-accelerated stream processing.
- **NVIDIA Triton Inference Server:** Handles log classification (Event Type, Priority, Security Flag) using an ensemble of ONNX models.
- **PyTorch (Autoencoders):** Implements the DFP anomaly detection logic. Each host gets a unique AutoEncoder profile.
- **MLflow:** Manages the versioning and persistence of host-specific DFP models.
- **Apache Kafka:** The high-speed ingestion and egress layer.
- **cuDF / CuPy:** Used for vectorized preprocessing and mathematical operations on the GPU.

---

## 🧠 Pipeline Architecture

### 1. Ingestion & Preprocessing (`GPUPreprocessingStage`)
- **Source Extraction:** Uses GPU-accelerated regex (`cudf.Series.str.extract`) to identify the source IP or hostname from raw syslog messages.
- **Hashing Vectorization:** Generates 8 content-based features by hashing log messages directly on the GPU. This allows the system to "understand" the difference between an SSH login, a sudo command, or a kernel panic without manual rules.
- **Temporal Compression:** Compresses logs into high-frequency spikes (microsecond spacing) to enhance the DFP model's sensitivity to volume anomalies.

### 2. Classification (`StringTritonInferenceStage`)
- Ensemble inference via Triton to determine:
    - `is_security_event` (Boolean flag)
    - `event_type` (e.g., Authentication, System, Network)
    - `priority` (e.g., Emergency, Alert, Critical)
- Uses **Zero-Copy Arrow-to-Numpy** transfers to avoid Python-level string overhead.

### 3. Digital Fingerprinting (`ManualInferenceStage`)
- **Dynamic Profile Loading:** Loads and promotes host-specific AutoEncoder models into GPU memory.
- **Multi-Model Inference:** Performs batched inference across hundreds of different host profiles simultaneously using a custom GPU-vectorized lookup and normalization engine.
- **Anomaly Attribution:** Identifies the specific cause of an anomaly (e.g., "Log Content Anomaly", "Volume Anomaly") based on which feature had the highest reconstruction error.

### 4. Continuous Learning (`AsyncTrainingStage`)
- Hosts are automatically identified and queued for training once they accumulate enough data.
- Models are trained in the background and promoted to the live inference engine without stopping the pipeline.

---

## 🚀 Performance Optimizations
- **Batched Sink:** The Kafka sink uses a consolidated buffer and high-speed JSON serialization to maximize output throughput.
- **Memory Management:** Uses **RMM (RAPIDS Memory Manager)** to prevent memory fragmentation and manage GPU memory pools effectively.
- **Thread Orchestration:** Uses a `ThreadPoolExecutor` to handle model training without blocking the main inference path.

---

## 📈 Feature Definition
The DFP model currently learns from **12 features**:
1. `feature_log_count`: Log volume frequency.
2. `feature_priority`: Syslog priority level.
3. `feature_event_type`: Classified category.
4. `feature_is_security_event`: Security relevance flag.
5. `feature_log_v0` - `feature_log_v7`: 8 hashing-based features representing log message content patterns.
