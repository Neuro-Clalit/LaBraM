# Data-preparation image for AWS Batch: EDF -> per-recording npy (make_TUH_npy).
# deploy.sh builds from a minimal context holding only this file, requirements.txt
# and the converter's three source files (it falls back to flat imports).
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY make_TUH_npy.py make_TUAB_npy.py tuh_metadata.py /app/
ENTRYPOINT ["python", "/app/make_TUH_npy.py"]
