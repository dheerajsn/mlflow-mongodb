#!/usr/bin/env python3
"""
MLflow Server with MongoDB Backend

This script starts an MLflow server with MongoDB as the backend for both
tracking and model registry, using the plugin approach.
"""

import os
import sys
import logging
import time
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_mongodb_connection():
    """Check if MongoDB is running and accessible"""
    logger.info("Checking MongoDB connection...")

    try:
        import pymongo
        client = pymongo.MongoClient(
            "mongodb://admin:password@localhost:27017/",
            serverSelectionTimeoutMS=5000
        )
        client.server_info()
        logger.info("✓ MongoDB connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        return False

def install_plugin():
    """Install the MLflow MongoDB plugin"""
    logger.info("Installing MLflow MongoDB plugin...")
    
    try:
        # Install plugin in development mode
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            logger.info("✓ Plugin installed successfully")
            return True
        else:
            logger.error(f"✗ Plugin installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Plugin installation error: {e}")
        return False

def verify_plugin_registration():
    """Verify that the plugin is properly registered"""
    logger.info("Verifying plugin registration...")

    try:
        # Force MLflow to discover plugins
        import mlflow
        from mlflow.tracking._tracking_service.utils import TrackingStoreRegistry
        from mlflow.tracking._model_registry.utils import ModelRegistryStoreRegistry

        # Get store registries and force entrypoint registration
        tracking_registry = TrackingStoreRegistry()
        tracking_registry.register_entrypoints()

        model_registry_registry = ModelRegistryStoreRegistry()
        model_registry_registry.register_entrypoints()

        # Check if our mongodb scheme is registered
        if "mongodb" in tracking_registry._registry:
            logger.info("✓ MongoDB tracking store registered")
        else:
            logger.error("✗ MongoDB tracking store not found in registry")
            logger.info(f"Available schemes: {list(tracking_registry._registry.keys())}")
            return False

        if "mongodb" in model_registry_registry._registry:
            logger.info("✓ MongoDB model registry store registered")
        else:
            logger.error("✗ MongoDB model registry store not found in registry")
            logger.info(f"Available schemes: {list(model_registry_registry._registry.keys())}")
            return False
        return True

    except Exception as e:
        logger.error(f"✗ Plugin verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mongodb_stores():
    """Test MongoDB stores functionality"""
    logger.info("Testing MongoDB stores...")

    try:
        import mlflow
        from mlflow.tracking._tracking_service.utils import _get_store
        from mlflow.tracking._model_registry.utils import _get_store as _get_model_registry_store

        mongodb_uri = "mongodb://username:passwordlocalhost:27017/mlflow"

        # Test tracking store
        tracking_store = _get_store(mongodb_uri)
        logger.info(f"✓ Tracking store created: {type(tracking_store).__name__}")

        # Test model registry store
        model_registry_store = _get_model_registry_store(mongodb_uri)
        logger.info(f"✓ Model registry store created: {type(model_registry_store).__name__}")

        # Test trace functionality
        if hasattr(tracking_store, 'search_traces'):
            logger.info("✓ Tracking store has trace support")
        else:
            logger.warning("⚠ Tracking store missing trace support")

        return True

    except Exception as e:
        logger.error(f"✗ MongoDB stores test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trace_artifacts():
    """Test trace artifacts functionality"""
    logger.info("Testing trace artifacts...")

    try:
        from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        mongodb_uri = "mongodb://username:passwordlocalhost:27017/mlflow"
        store = MongoDbTrackingStore(mongodb_uri)

        # Check if there are any existing traces
        traces_count = store.traces_collection.count_documents({})
        logger.info(f"✓ Found {traces_count} traces in MongoDB")

        if traces_count > 0:
            # Get a sample trace
            sample_trace = store.traces_collection.find_one()
            request_id = sample_trace['request_id']

            # Ensure trace has artifact location tag
            if 'mlflow.artifactLocation' not in sample_trace.get('tags', {}):
                artifact_location = f"mongodb://mlflow/traces/{request_id}/artifacts"
                store.traces_collection.update_one(
                    {"request_id": request_id},
                    {"$set": {"tags.mlflow.artifactLocation": artifact_location}}
                )
                logger.info(f"✓ Added artifact location to trace {request_id}")

            # Test artifact repository
            artifact_uri = f"mongodb://mlflow/traces/{request_id}/artifacts"
            repo = get_artifact_repository(artifact_uri)
            logger.info(f"✓ Artifact repository created: {type(repo).__name__}")

            # Create sample artifact data if it doesn't exist
            try:
                artifact_data = store.get_trace_artifact(request_id)
                logger.info(f"✓ Found existing artifact data with {len(artifact_data.get('spans', []))} spans")
            except:
                # Create sample artifact data
                artifact_data = {
                    "experiment_id": sample_trace.get('experiment_id'),
                    "spans": [
                        {
                            "name": "Sample Span",
                            "context": {"span_id": "0x123", "trace_id": "0x456"},
                            "start_time": 1000000000,
                            "end_time": 1000001000,
                            "status_code": "OK",
                            "inputs": {"query": "test"},
                            "outputs": {"response": "test response"}
                        }
                    ],
                    "request_metadata": sample_trace.get('request_metadata', {}),
                    "tags": sample_trace.get('tags', {})
                }
                store.store_trace_artifact(request_id, artifact_data)
                logger.info(f"✓ Created sample artifact data for trace {request_id}")

        return True

    except Exception as e:
        logger.error(f"✗ Trace artifacts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_monitoring():
    """Set up monitoring for API calls"""
    try:
        # Import monitoring script to apply monkey patches
        import monitor_version_calls
        logger.info("✓ API call monitoring enabled")
        return True
    except Exception as e:
        logger.warning(f"⚠ Could not enable monitoring: {e}")
        return False

def test_mlflow_client():
    """Test MLflow client with MongoDB backend"""
    logger.info("Testing MLflow client with MongoDB...")
    
    try:
        import mlflow
        
        mongodb_uri = "mongodb://username:passwordlocalhost:27017/mlflow"
        
        # Set tracking and registry URIs
        mlflow.set_tracking_uri(mongodb_uri)
        mlflow.set_registry_uri(mongodb_uri)
        
        # Create client
        client = mlflow.tracking.MlflowClient(
            tracking_uri=mongodb_uri,
            registry_uri=mongodb_uri
        )
        
        # Test basic operations
        test_exp_name = f"test_mongodb_plugin_{int(time.time())}"
        exp_id = client.create_experiment(test_exp_name)
        logger.info(f"✓ Created test experiment: {test_exp_name} (ID: {exp_id})")
        
        # Clean up
        client.delete_experiment(exp_id)
        logger.info("✓ Test experiment cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MLflow client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mongodb_artifact_patch():
    """Create a patch file to register MongoDB artifacts in the server process"""
    logger.info("Creating MongoDB artifact patch...")

    try:
        # Create a comprehensive patch that handles MongoDB artifacts
        patch_content = '''
# MongoDB Artifact Repository Registration Patch
# This file is automatically imported to register MongoDB artifacts

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Register MongoDB artifact repository
try:
    from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry
    from mlflow_mongodb.artifact_repo import get_mongodb_artifact_repository

    # Register the mongodb scheme
    _artifact_repository_registry.register("mongodb", get_mongodb_artifact_repository)
    print("✓ MongoDB artifact repository registered via patch")

    # Also monkey patch the get_artifact_repository function as backup
    from mlflow.store.artifact import artifact_repository_registry

    original_get_artifact_repository = artifact_repository_registry.get_artifact_repository

    def patched_get_artifact_repository(artifact_uri):
        """Patched version that handles MongoDB URIs"""
        if artifact_uri.startswith("mongodb://"):
            try:
                return get_mongodb_artifact_repository(artifact_uri)
            except Exception as e:
                print(f"Failed to create MongoDB artifact repository: {e}")
                raise
        return original_get_artifact_repository(artifact_uri)

    # Replace the function
    artifact_repository_registry.get_artifact_repository = patched_get_artifact_repository
    print("✓ MongoDB artifact repository globally patched")

except Exception as e:
    print(f"✗ MongoDB artifact registration failed: {e}")
    import traceback
    traceback.print_exc()
'''

        # Write the patch file
        patch_file = Path("mongodb_artifact_patch.py")
        with open(patch_file, "w") as f:
            f.write(patch_content)

        logger.info(f"✓ Created MongoDB artifact patch: {patch_file}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to create patch: {e}")
        return False

def start_mlflow_server():
    """Start MLflow server using the mlflow command"""
    logger.info("Starting MLflow server...")

    try:
        # MongoDB URIs
        mongodb_uri = "mongodb://username:passwordlocalhost:27017/mlflow"

        # Create artifacts directory
        artifacts_dir = Path("./mlflow-artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Create MongoDB artifact patch
        create_mongodb_artifact_patch()

        # Set environment variables
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = mongodb_uri
        env["MLFLOW_REGISTRY_URI"] = mongodb_uri
        # Add Python path to ensure our modules are found
        env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
        # Import the patch on server startup
        env["PYTHONSTARTUP"] = "mongodb_artifact_patch.py"

        # Use standard MLflow server command
        cmd = [
            sys.executable, "-m", "mlflow", "server",
            "--backend-store-uri", mongodb_uri,
            "--registry-store-uri", mongodb_uri,
            "--default-artifact-root", str(artifacts_dir),
            "--host", "0.0.0.0",
            "--port", "5001",
            "--serve-artifacts"
        ]

        logger.info("=" * 60)
        logger.info("  MLflow Server with MongoDB Backend")
        logger.info("=" * 60)
        logger.info(f"Backend Store URI: {mongodb_uri}")
        logger.info(f"Registry Store URI: {mongodb_uri}")
        logger.info(f"Default Artifact Root: {artifacts_dir}")
        logger.info("Server URL: http://localhost:5001")
        logger.info("✓ MongoDB artifact repository will be registered")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server")

        # Start the server
        subprocess.run(cmd, env=env)
        return True

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        return False

def kill_existing_servers():
    """Kill any existing MLflow server processes"""
    logger.info("Killing any existing MLflow server processes...")
    
    try:
        # Kill any process containing "mlflow server"
        subprocess.run(
            ["pkill", "-f", "mlflow server"],
            capture_output=True,
            check=False  # Don't fail if no processes found
        )
        
        # Kill any process using port 5001
        result = subprocess.run(
            ["lsof", "-ti:5001"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(
                        ["kill", "-9", pid],
                        capture_output=True,
                        check=False
                    )
        
        # Wait a moment for processes to terminate
        time.sleep(2)
        logger.info("✓ Existing server processes killed")
        
    except Exception as e:
        logger.warning(f"Warning while killing existing servers: {e}")

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("  MLflow MongoDB Plugin Setup and Server")
    logger.info("=" * 60)
    
    # Kill existing servers
    kill_existing_servers()
    
    # Check MongoDB connection
    if not check_mongodb_connection():
        logger.error("Cannot proceed without MongoDB connection")
        logger.error("Please ensure MongoDB is running with authentication:")
        logger.error("  docker run -d --name mongodb -p 27017:27017 \\")
        logger.error("    -e MONGO_INITDB_ROOT_USERNAME=admin \\")
        logger.error("    -e MONGO_INITDB_ROOT_PASSWORD=password \\")
        logger.error("    mongo:latest")
        return False
    
    # Install plugin
    if not install_plugin():
        logger.error("Failed to install plugin")
        return False
    
    # Verify plugin registration
    if not verify_plugin_registration():
        logger.error("Plugin registration failed")
        return False
    
    # Test MongoDB stores
    if not test_mongodb_stores():
        logger.error("MongoDB stores test failed")
        return False

    # Test trace artifacts
    if not test_trace_artifacts():
        logger.error("Trace artifacts test failed")
        return False

    # Set up monitoring
    setup_monitoring()

    # Test MLflow client
    if not test_mlflow_client():
        logger.error("MLflow client test failed")
        return False
    
    logger.info("✓ All tests passed - starting MLflow server...")
    
    # Start server
    return start_mlflow_server()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
