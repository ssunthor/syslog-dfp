# Syslog DFP (Digital Fingerprinting) Pipeline

**Patent Pending -- Copyright 2026 GTT Communications**
**DO NOT SHARE THIS CODE WITH ANYONE WITHOUT APPROVAL**

A high-performance, GPU-accelerated syslog analysis and anomaly detection system built on **NVIDIA Morpheus**. This pipeline uses the Digital Fingerprinting (DFP) pattern to learn the unique behavioral "fingerprint" of every host in a network and detect security threats or operational failures in real-time.

---

## 🚀 Overview

The pipeline processes raw syslog messages at extreme throughput, performing deep classification via **NVIDIA Triton** and behavioral anomaly detection via **PyTorch AutoEncoders**. Unlike traditional rule-based systems, this pipeline learns what is "normal" for each specific host and flags deviations across temporal, identity, and content-based dimensions.

---

## 🧠 Anomaly Detection (DFP) Features

The system learns from a **19-dimensional feature vector** for every host. When an anomaly is detected, the pipeline identifies the specific attribute that contributed most to the reconstruction error, providing a human-readable `anomaly_cause`.

### 1. Behavioral & Volume
*   **Volume Anomaly:** Triggered when the log frequency (EPS) for a host significantly deviates from its historical average.
*   **Severity Spike:** Detects when a host that usually sends `INFO` or `NOTICE` logs suddenly starts emitting `CRITICAL`, `ALERT`, or `EMERGENCY` messages.
*   **Unusual Event:** Identified when the classified `event_type` (e.g., Authentication, Network, System) is rare for that specific host.
*   **Security Flag:** Triggered when Triton classifies a log as having a high probability of being a security-relevant event for a host that rarely generates them.

### 2. Temporal (Day & Time)
*   **Time Anomaly (Hour):** Detects execution at unusual hours (e.g., a backup script running at 2:00 PM instead of 2:00 AM). Uses cyclical sine/cosine encoding to ensure the model understands that 23:59 and 00:01 are close together.
*   **Time Anomaly (Day):** Flags activity on unusual days of the week (e.g., administrative commands executed on a Sunday for a host that is typically only managed during weekdays).

### 3. Identity & Metadata
*   **User Anomaly:** Triggered when a log contains a username (e.g., `admin`, `root`, `service_account`) that is not typically associated with that host.
*   **Process Anomaly:** Detects activity from an unusual application or service (e.g., `nc` or `nmap` logging on a database server where only `postgres` and `systemd` are normal).
*   **Message Length Anomaly:** Flags logs that are unusually long or short, which can signify buffer overflows, large command-line arguments, or truncated logs.

### 4. Content & Pattern (Semantic Hashing)
*   **Log Content Anomaly:** The system maintains 8 independent hashing features (`feature_log_v0` through `feature_log_v7`). These capture the "shape" and vocabulary of the log message itself. This allows the model to detect new types of logs or changes in log structure even if the username or process remains the same.

### 5. "This is Normal" (User Feedback & Fine-Tuning)
The system supports **Online Fine-Tuning** through user feedback. If a legitimate (but rare) activity is flagged as an anomaly, users can mark it as "normal."
- **Immediate Nudging:** The specific host's model is immediately fine-tuned in the background with the user-provided sample.
- **Dynamic Thresholding:** The model's internal representation of "normal" for that host is adjusted, causing subsequent similar logs to receive a low anomaly score.
- **Long-term Adaptation:** Feedback samples are persisted and automatically merged into all future full training cycles for that host.

---

## 🔌 API Documentation

The pipeline exposes a FastAPI management interface (default: `13005`) for health monitoring and user feedback.

### `GET /health`
Returns the current status of the pipeline, last activity timestamp, and overall system health.

### `POST /feedback`
Submits a "This is Normal" override for a specific log entry.
- **Payload:**
  ```json
  {
    "source": "host_id",
    "features": {
      "feature_log_v0": 0.123,
      "feature_priority": 0.5,
      ...
    }
  }
  ```
- **Action:** Triggers an immediate background fine-tuning of the host-specific AutoEncoder model and persists the feedback for future training cycles.

---

## 🛠 Technology Stack

- **NVIDIA Morpheus:** Core pipeline orchestration.
- **NVIDIA Triton Inference Server:** Ensemble log classification (is_security, event_type, priority).
- **PyTorch (AutoEncoders):** Individualized host behavioral models.
- **MLflow:** Model versioning and lifecycle management.
- **Apache Kafka:** High-speed streaming ingestion and egress.
- **cuDF & CuPy:** Zero-copy, GPU-accelerated feature engineering and preprocessing.
- **FastAPI:** Real-time health monitoring and pipeline status.

---

## 🏗 Pipeline Architecture

1.  **Ingestion:** Consumes raw syslog from Kafka.
2.  **GPU Preprocessing:** Vectorized regex extraction and cyclical temporal encoding via `cudf`.
3.  **Triton Classification:** Multi-model ensemble determines the semantic intent of the log.
4.  **Accumulator:** High-speed GPU batching to optimize DFP inference throughput.
5.  **Digital Fingerprinting (Inference):** Parallelized AutoEncoder reconstruction across hundreds of host profiles simultaneously in VRAM.
6.  **Continuous Learning:** Background training of host profiles using `AsyncTrainingStage`, automatically promoting new models into the live inference engine.
7.  **Egress:** Enriched anomalies (with scores and causes) are pushed to a downstream Kafka topic for visualization or alerting.

---

## 📊 Performance Optimizations

- **Zero-Copy Philosophy:** Data remains in GPU memory (VRAM) from ingestion through inference.
- **RMM (RAPIDS Memory Manager):** Pooled memory allocation to prevent fragmentation.
- **Multi-Model Lookup:** Custom GPU-vectorized hash lookup to map logs to host-specific models in `O(1)` time.
- **Cyclical Encoding:** Ensures temporal continuity for periodic behavioral patterns.
