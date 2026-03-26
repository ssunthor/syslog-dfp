FROM nvcr.io/nvidia/morpheus/morpheus:25.06-runtime

WORKDIR /app

# FIX: Added '-y' to automatically accept the installation
# FIX: Removed version pin for scikit-learn
RUN conda install -y -c nvidia -c rapidsai -c conda-forge \
    morpheus-dfp=25.06 \
    mlflow \
    && conda clean --all -f -y

COPY morpheus_app.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "morpheus_app.py"]