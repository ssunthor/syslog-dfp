# Morpheus Inference Pipeline

**Author:** Scott Chrestman
**Copyright:** 2025 GTT Communications, Inc.

This directory contains the NVIDIA Morpheus pipeline for real-time log analysis and anomaly detection.

## Overview

This pipeline consumes log messages from a Kafka topic, performs inference using a Triton Inference Server, and publishes the enriched data back to another Kafka topic. The pipeline is designed to be run in a Docker container and is orchestrated using Docker Compose.

The core components are:
-   `morpheus_app.py`: The main Morpheus pipeline script.
-   `Dockerfile`: Defines the Docker image for the Morpheus application.
-   `docker-compose.yaml`: Orchestrates the Morpheus application and the Triton Inference Server.

The pipeline performs the following steps:
1.  Reads log messages from the `syslog-test` Kafka topic.
2.  Sends the log messages to a Triton Inference Server for classification.
3.  The Triton server uses the trained models to predict the event type, priority, and whether it's a security event.
4.  The pipeline decodes the model's output (integer labels) back into human-readable strings (e.g., "High", "Authentication").
5.  The enriched messages are published to the `morpheus-output` Kafka topic.

## Setup

### Prerequisites
-   Docker and Docker Compose
-   NVIDIA GPU with CUDA drivers

### 1. Build and Run the Pipeline

The `docker-compose.yaml` file is configured to build and run both the Triton Inference Server and the Morpheus pipeline.

From the `morpheus_pipeline` directory, run:

```bash
docker-compose up --build
```

This command will:
-   Build the custom Triton server image (if `Dockerfile.triton` is present).
-   Build the Morpheus application image.
-   Start both the `triton-server` and `morpheus-app` containers.

### 2. Configuration

The pipeline's configuration is managed through environment variables in the `docker-compose.yaml` file. You may need to adjust the following variables to match your environment:

-   `KAFKA_BROKER`: The address of your Kafka broker.
-   `INPUT_TOPIC`: The Kafka topic to consume messages from.
-   `OUTPUT_TOPIC`: The Kafka topic to publish enriched messages to.
-   `TRITON_SERVER_URL`: The URL of the Triton Inference Server.

**Note:** The `models` directory from the parent directory is mounted as a volume in both the Triton and Morpheus containers. This allows both services to access the trained models and label encoders.
