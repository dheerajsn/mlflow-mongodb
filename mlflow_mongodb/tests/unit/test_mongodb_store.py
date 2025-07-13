#!/usr/bin/env python3
"""
Unit tests for MongoDB Store utilities and components
"""

import pytest
import unittest.mock as mock
from mlflow_mongodb.db_utils import MongoDbUtils, get_current_time_millis, parse_mongodb_uri



class TestMongoDbUtils:
    """Unit tests for MongoDB utilities"""

    def test_get_current_time_millis(self):
        """Test current time in milliseconds"""
        timestamp = get_current_time_millis()
        assert isinstance(timestamp, int)
        assert timestamp > 0

    def test_parse_mongodb_uri_basic(self):
        """Test basic MongoDB URI parsing"""
        uri = "mongodb://localhost:27017/testdb"
        result = parse_mongodb_uri(uri)

        assert result["database_name"] == "testdb"
        assert result["collection_prefix"] == "mlflow"
        assert "mongodb://localhost:27017" in result["connection_string"]

    def test_parse_mongodb_uri_with_auth(self):
        """Test MongoDB URI parsing with authentication"""
        uri = "mongodb://user:pass@localhost:27017/testdb"
        result = parse_mongodb_uri(uri)

        assert result["database_name"] == "testdb"
        assert "user:pass" in result["connection_string"]

    def test_parse_mongodb_uri_with_prefix(self):
        """Test MongoDB URI parsing with collection prefix"""
        uri = "mongodb://localhost:27017/testdb#custom"
        result = parse_mongodb_uri(uri)

        assert result["database_name"] == "testdb"
        assert result["collection_prefix"] == "custom"

    def test_parse_mongodb_uri_defaults(self):
        """Test MongoDB URI parsing with defaults"""
        uri = "mongodb://localhost:27017"
        result = parse_mongodb_uri(uri)

        assert result["database_name"] == "mlflow"  # default
        assert result["collection_prefix"] == "mlflow"  # default

    def test_mongodb_utils_parse_uri(self):
        """Test MongoDbUtils URI parsing"""
        uri = "mongodb://localhost:27017/testdb"
        connection_string, database_name, collection_prefix = MongoDbUtils.parse_mongodb_uri(uri)

        assert database_name == "testdb"
        assert collection_prefix == "mlflow"
        assert "mongodb://localhost:27017" in connection_string

    def test_invalid_scheme(self):
        """Test invalid URI scheme handling"""
        with pytest.raises(ValueError, match="Invalid MongoDB URI scheme"):
            MongoDbUtils.parse_mongodb_uri("invalid://localhost:27017/testdb")

    @mock.patch('mlflow_mongodb.db_utils.MongoClient')
    def test_create_client(self, mock_mongo_client):
        """Test MongoDB client creation"""
        mock_client = mock.MagicMock()
        mock_mongo_client.return_value = mock_client

        client = MongoDbUtils.create_client("mongodb://localhost:27017", "testdb")

        mock_mongo_client.assert_called_once()
        assert client == mock_client


class TestInfiniteLoopPrevention:
    """Test infinite loop prevention in model registry"""

    @mock.patch('mlflow_mongodb.db_utils.MongoClient')
    def test_prompt_version_infinite_loop_prevention(self, mock_mongo_client):
        """Test that get_prompt_version doesn't cause infinite loop"""
        import sys
        import os
        # Add current directory to path to use local version
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_registry.mongodb_store import MongoDbModelRegistryStore
        from mlflow.exceptions import MlflowException

        mock_client = mock.MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_client.admin.command.return_value = True

        store = MongoDbModelRegistryStore("mongodb://localhost:27017/testdb")

        # Mock the collections
        store.model_versions_collection = mock.MagicMock()
        store.registered_model_aliases_collection = mock.MagicMock()
        store.registered_model_tags_collection = mock.MagicMock()

        # Test case: invalid version from alias should not cause infinite loop
        store.registered_model_aliases_collection.find_one.return_value = {
            "name": "test_prompt",
            "alias": "latest",
            "version": "invalid_version"  # This would cause infinite loop before fix
        }

        # Should raise exception, not infinite loop
        # First test: check that our method is being called
        assert hasattr(store, 'get_prompt_version_by_alias')

        # Test the infinite loop prevention
        with pytest.raises(MlflowException, match="Invalid version"):
            store.get_prompt_version_by_alias("test_prompt", "latest")


if __name__ == "__main__":
    pytest.main([__file__])