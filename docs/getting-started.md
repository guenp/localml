# Getting Started

## Requirements

- Python 3.11+
- Docker Compose for the local control-plane services
- `uv` for development

## Install

```sh
pip install localml
```

For local development:

```sh
uv sync
```

## Run the Local Stack

```sh
cp .env.example .env
docker compose up -d
```

The API is available at `http://localhost:8000`, and the generated FastAPI docs are at
`http://localhost:8000/docs`.

## Preview Documentation

```sh
uv run zensical serve
```
