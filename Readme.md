# mlflow-mongodb

A custom MLflow tracking store backend using MongoDB instead of a relational database (SQLite/Postgres/MySQL) as the metadata store.

## Overview

This project implements an MLflow `AbstractStore` backed by MongoDB, allowing MLflow experiment tracking, runs, and metadata to be persisted in a MongoDB instance rather than a traditional SQL backend.

## Requirements

- Python 3.x
- MongoDB instance (local or remote)
- MLflow

## Installation

```bash
pip install -r requirements.txt
```

## Setup

```bash
./create.sh
```

*(brief note on what create.sh does — e.g. sets up the Mongo collections/indexes)*

## Usage

```python
import mlflow

mlflow.set_tracking_uri("mongodb://<host>:<port>/<db_name>")
# your MLflow tracking code as usual
```

## Project Structure

- `mlflow_mongodb/` — core package implementing the MongoDB-backed tracking store
- `docs/` — documentation
- `scripts/build/` — build scripts
- `create.sh` — setup script

## Status

*(e.g. "Prototype / actively developed / seeking feedback")*

## License

*(add a license, e.g. MIT, Apache 2.0 — or note internal-use-only if applicable)*