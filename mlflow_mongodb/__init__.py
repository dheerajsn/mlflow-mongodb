"""
MongoDB Store Implementation for MLflow 3.0+

This package provides MongoDB-based storage backends for MLflow tracking and model registry.
"""

__all__ = ["MongoDbTrackingStore", "MongoDbModelRegistryStore"]

def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name == "MongoDbTrackingStore":
        from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
        return MongoDbTrackingStore
    elif name == "MongoDbModelRegistryStore":
        from mlflow_mongodb.model_registry.mongodb_store import MongoDbModelRegistryStore
        return MongoDbModelRegistryStore
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
