"""
MongoDB Database Utilities for MLflow Stores
"""

from typing import Optional, Dict, Any, List
import time
import pymongo
from pymongo import MongoClient, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database
from urllib.parse import urlparse


def get_current_time_millis():
    """Get current time in milliseconds"""
    return int(time.time() * 1000)


def parse_mongodb_uri(uri: str) -> Dict[str, str]:
    """Parse MongoDB URI and return components"""
    utils = MongoDbUtils()
    connection_string, database_name, collection_prefix = utils.parse_mongodb_uri(uri)
    return {
        "connection_string": connection_string,
        "database_name": database_name,
        "collection_prefix": collection_prefix
    }


class MongoDbUtils:
    """Utilities for MongoDB operations in MLflow stores"""
    
    @staticmethod
    def parse_mongodb_uri(uri: str) -> tuple:
        """
        Parse MongoDB URI to extract connection details.
        
        Args:
            uri: MongoDB URI in format mongodb://[username:password@]host[:port]/database[?options][#collection_prefix]
            
        Returns:
            tuple of (connection_string, database_name, collection_prefix)
        """
        parsed = urlparse(uri)
        
        if parsed.scheme not in ["mongodb", "mongodb+srv"]:
            raise ValueError(f"Invalid MongoDB URI scheme: {parsed.scheme}. Expected 'mongodb' or 'mongodb+srv'")
        
        # Extract database name from path
        database_name = parsed.path.lstrip('/') if parsed.path else 'mlflow'
        
        # Reconstruct connection string with authentication if present
        if parsed.username and parsed.password:
            # Include authentication in connection string
            connection_string = f"{parsed.scheme}://{parsed.username}:{parsed.password}@{parsed.hostname}"
            if parsed.port:
                connection_string += f":{parsed.port}"

            # Add authSource to query parameters if not present
            query_params = parsed.query
            if query_params and "authSource" not in query_params:
                query_params += "&authSource=admin"
            elif not query_params:
                query_params = "authSource=admin"

            if query_params:
                connection_string += f"?{query_params}"
        else:
            # No authentication
            connection_string = f"{parsed.scheme}://{parsed.netloc}"
            # Add query parameters if present
            if parsed.query:
                connection_string += f"?{parsed.query}"
        
        # Use collection prefix from fragment or default
        collection_prefix = parsed.fragment or "mlflow"
        
        return connection_string, database_name, collection_prefix
    
    @staticmethod
    def create_client(connection_string: str, database_name: str = None) -> MongoClient:
        """Create MongoDB client with proper configuration"""
        # Parse the connection string to check if authentication is included
        from urllib.parse import urlparse
        parsed = urlparse(connection_string)

        client_options = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 10000,
            "socketTimeoutMS": 10000,
            "maxPoolSize": 50,
            "minPoolSize": 5,
            "maxIdleTimeMS": 30000,
            "waitQueueTimeoutMS": 10000,
            "journal": True,
            "w": "majority",
            "wtimeoutMS": 10000,
        }

        # Add authentication source if credentials are present
        if parsed.username and parsed.password:
            # Check if authSource is already in query params
            if "authSource" not in connection_string:
                client_options["authSource"] = "admin"

        return MongoClient(connection_string, **client_options)
    
    @staticmethod
    def ensure_indexes(collection: Collection, indexes: List[IndexModel]) -> None:
        """Ensure indexes exist on a collection"""
        if indexes:
            try:
                collection.create_indexes(indexes)
            except Exception as e:
                # Log warning but don't fail - indexes are for performance
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create indexes on {collection.name}: {e}")
                # Re-raise authentication errors as they indicate a configuration problem
                if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                    raise
    
    @staticmethod
    def create_indexes_for_experiments(collection: Collection) -> None:
        """Create indexes for experiments collection"""
        indexes = [
            IndexModel([("experiment_id", pymongo.ASCENDING)], unique=True),
            IndexModel([("name", pymongo.ASCENDING)], unique=True),
            IndexModel([("lifecycle_stage", pymongo.ASCENDING)]),
            IndexModel([("creation_time", pymongo.DESCENDING)]),
            IndexModel([("last_update_time", pymongo.DESCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_runs(collection: Collection) -> None:
        """Create indexes for runs collection"""
        indexes = [
            IndexModel([("run_uuid", pymongo.ASCENDING)], unique=True),
            IndexModel([("experiment_id", pymongo.ASCENDING)]),
            IndexModel([("status", pymongo.ASCENDING)]),
            IndexModel([("lifecycle_stage", pymongo.ASCENDING)]),
            IndexModel([("start_time", pymongo.DESCENDING)]),
            IndexModel([("end_time", pymongo.DESCENDING)]),
            IndexModel([("experiment_id", pymongo.ASCENDING), ("start_time", pymongo.DESCENDING)]),
            IndexModel([("user_id", pymongo.ASCENDING)]),
            IndexModel([("name", pymongo.ASCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_metrics(collection: Collection) -> None:
        """Create indexes for metrics collection"""
        indexes = [
            IndexModel([("run_uuid", pymongo.ASCENDING), ("key", pymongo.ASCENDING)]),
            IndexModel([("run_uuid", pymongo.ASCENDING), ("key", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)]),
            IndexModel([("run_uuid", pymongo.ASCENDING), ("key", pymongo.ASCENDING), ("step", pymongo.ASCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_params(collection: Collection) -> None:
        """Create indexes for params collection"""
        indexes = [
            IndexModel([("run_uuid", pymongo.ASCENDING), ("key", pymongo.ASCENDING)], unique=True),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_tags(collection: Collection) -> None:
        """Create indexes for tags collection"""
        indexes = [
            IndexModel([("run_uuid", pymongo.ASCENDING), ("key", pymongo.ASCENDING)], unique=True),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_registered_models(collection: Collection) -> None:
        """Create indexes for registered models collection"""
        indexes = [
            IndexModel([("name", pymongo.ASCENDING)], unique=True),
            IndexModel([("creation_time", pymongo.DESCENDING)]),
            IndexModel([("last_updated_time", pymongo.DESCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
    
    @staticmethod
    def create_indexes_for_model_versions(collection: Collection) -> None:
        """Create indexes for model versions collection"""
        indexes = [
            IndexModel([("name", pymongo.ASCENDING), ("version", pymongo.ASCENDING)], unique=True),
            IndexModel([("name", pymongo.ASCENDING)]),
            IndexModel([("run_id", pymongo.ASCENDING)]),
            IndexModel([("current_stage", pymongo.ASCENDING)]),
            IndexModel([("creation_time", pymongo.DESCENDING)]),
            IndexModel([("last_updated_time", pymongo.DESCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(collection, indexes)
