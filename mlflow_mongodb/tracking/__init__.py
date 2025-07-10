"""
MongoDB Tracking Store Module
"""

__all__ = ["MongoDbTrackingStore"]

def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name == "MongoDbTrackingStore":
        from .mongodb_store import MongoDbTrackingStore
        return MongoDbTrackingStore
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
