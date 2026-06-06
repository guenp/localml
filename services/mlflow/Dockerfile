FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow>=2.12 psycopg[binary]>=3.1 boto3>=1.34

EXPOSE 5000

# Backend store: Postgres. Artifact store: MinIO (S3-compatible).
# Endpoint/credentials are provided via env (see .env.example / docker-compose.yml).
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin

CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri ${DATABASE_URL} \
  --artifacts-destination s3://${MINIO_BUCKET}"]
