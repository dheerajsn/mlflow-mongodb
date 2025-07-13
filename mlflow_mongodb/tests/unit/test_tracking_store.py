#!/usr/bin/env python3
"""Unit tests for MongoDB tracking store"""

import pytest
import unittest.mock as mock
from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
from mlflow_mongodb.db_utils import MongoDbUtils


class TestMongoDbTrackingStore:
    """Unit tests for MongoDB tracking store"""

    def test_uri_parsing(self):
        """Test MongoDB URI parsing"""
        uri = "mongodb://user:pass@localhost:27017/testdb"
        connection_string, database_name, collection_prefix = MongoDbUtils.parse_mongodb_uri(uri)

        assert "mongodb://user:pass@localhost:27017" in connection_string
        assert database_name == "testdb"
        assert collection_prefix == "mlflow"

    def test_uri_parsing_with_prefix(self):
        """Test MongoDB URI parsing with collection prefix"""
        uri = "mongodb://localhost:27017/testdb#custom_prefix"
        connection_string, database_name, collection_prefix = MongoDbUtils.parse_mongodb_uri(uri)

        assert "mongodb://localhost:27017" in connection_string
        assert database_name == "testdb"
        assert collection_prefix == "custom_prefix"


    @mock.patch('mlflow_mongodb.db_utils.MongoClient')
    def test_store_initialization(self, mock_mongo_client):
        """Test MongoDB tracking store initialization"""
        mock_client = mock.MagicMock()
        mock_mongo_client.return_value = mock_client

        # Mock the ping command to avoid connection errors
        mock_client.admin.command.return_value = True

        uri = "mongodb://localhost:27017/testdb"
        store = MongoDbTrackingStore(uri)

        assert store.database_name == "testdb"
        assert store.collection_prefix == "mlflow"
        mock_mongo_client.assert_called_once()

    @mock.patch('mlflow_mongodb.db_utils.MongoClient')
    def test_collection_initialization(self, mock_mongo_client):
        """Test collection initialization"""
        mock_client = mock.MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_client.admin.command.return_value = True

        store = MongoDbTrackingStore("mongodb://localhost:27017/testdb")

        # Check that collections are properly initialized
        assert hasattr(store, 'experiments_collection')
        assert hasattr(store, 'runs_collection')
        assert hasattr(store, 'metrics_collection')

    def test_invalid_uri(self):
        """Test invalid URI handling"""
        with pytest.raises(ValueError):
            MongoDbUtils.parse_mongodb_uri("invalid://uri")

    def test_default_values(self):
        """Test default values in URI parsing"""
        uri = "mongodb://localhost:27017"
        connection_string, database_name, collection_prefix = MongoDbUtils.parse_mongodb_uri(uri)

        assert database_name == "mlflow"  # default database
        assert collection_prefix == "mlflow"  # default prefix


if __name__ == "__main__":
    pytest.main([__file__])
