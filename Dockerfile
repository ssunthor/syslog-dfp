# Use NVIDIA Morpheus Base Image
FROM nvcr.io/nvidia/morpheus/morpheus:25.06-runtime AS base

WORKDIR /app

# Copy pipeline code
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "morpheus_app.py"]