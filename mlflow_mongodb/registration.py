"""
MongoDB Store Registration for MLflow

This module provides functions to register MongoDB stores with MLflow.
"""

from typing import Optional
from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
from mlflow_mongodb.model_registry.mongodb_store import MongoDbModelRegistryStore


def get_mongodb_tracking_store(store_uri: str, artifact_uri: Optional[str] = None) -> MongoDbTrackingStore:
    """
    Get MongoDB tracking store instance.
    
    Args:
        store_uri: MongoDB connection URI
        artifact_uri: Artifact store URI (optional)
        
    Returns:
        MongoDbTrackingStore instance
    """
    return MongoDbTrackingStore(store_uri, artifact_uri)


def get_mongodb_model_registry_store(store_uri: str, tracking_uri: Optional[str] = None) -> MongoDbModelRegistryStore:
    """
    Get MongoDB model registry store instance.
    
    Args:
        store_uri: MongoDB connection URI
        tracking_uri: Tracking store URI (optional)
        
    Returns:
        MongoDbModelRegistryStore instance
    """
    return MongoDbModelRegistryStore(store_uri, tracking_uri)


# Plugin entry points for MLflow store registration
def register_mongodb_stores():
    """Register MongoDB stores with MLflow (for plugin-based registration)"""
    try:
        # Import MLflow store registries
        from mlflow.tracking._tracking_service.utils import _tracking_store_registry
        from mlflow.tracking._model_registry.utils import _model_registry_store_registry
        
        # Register tracking store
        _tracking_store_registry.register("mongodb", get_mongodb_tracking_store)
        
        # Register model registry store
        _model_registry_store_registry.register("mongodb", get_mongodb_model_registry_store)
        
        print("MongoDB stores successfully registered with MLflow")
        
    except ImportError as e:
        print(f"Could not register MongoDB stores: {e}")
        print("Make sure MLflow is installed and accessible")
    except Exception as e:
        print(f"Error registering MongoDB stores: {e}")


# Auto-registration when module is imported
if __name__ != "__main__":
    register_mongodb_stores()
