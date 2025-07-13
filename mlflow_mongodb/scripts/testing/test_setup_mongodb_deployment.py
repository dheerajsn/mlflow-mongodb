#!/usr/bin/env python3
"""
Tests for setup_mongodb_deployment.py
"""

import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import sys

# Add current directory to path for imports
sys.path.insert(0, '.')

class TestSetupMongoDBDeployment(unittest.TestCase):
    """Test cases for setup_mongodb_deployment.py"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_check_system_requirements_success(self):
        """Test system requirements check with Python 3.8+"""
        from setup_mongodb_deployment import check_system_requirements
        
        # Should pass with current Python version (assuming 3.8+)
        result = check_system_requirements()
        self.assertTrue(result)
    
    @patch('sys.version_info', (3, 7, 0))
    def test_check_system_requirements_old_python(self):
        """Test system requirements check with old Python"""
        from setup_mongodb_deployment import check_system_requirements
        
        result = check_system_requirements()
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_check_system_requirements_mongodb_available(self, mock_run):
        """Test system requirements with MongoDB available"""
        from setup_mongodb_deployment import check_system_requirements
        
        # Mock MongoDB version command success
        mock_run.return_value = MagicMock(returncode=0)
        
        result = check_system_requirements()
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_check_system_requirements_mongodb_missing(self, mock_run):
        """Test system requirements with MongoDB missing"""
        from setup_mongodb_deployment import check_system_requirements
        
        # Mock MongoDB command not found
        mock_run.side_effect = FileNotFoundError("mongod not found")
        
        result = check_system_requirements()
        self.assertTrue(result)  # Should still pass, just warn
    
    @patch('pymongo.MongoClient')
    def test_setup_mongodb_database_success(self, mock_client):
        """Test successful MongoDB database setup"""
        from setup_mongodb_deployment import setup_mongodb_database
        
        # Mock MongoDB client and operations
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock database operations
        mock_db = MagicMock()
        mock_instance.mlflow = mock_db
        mock_db.list_collection_names.return_value = []
        mock_db.create_collection.return_value = None
        
        # Mock admin operations
        mock_admin = MagicMock()
        mock_instance.admin = mock_admin
        mock_admin.command.return_value = {"users": []}
        
        result = setup_mongodb_database()
        
        self.assertTrue(result)
        mock_instance.admin.command.assert_called()
        mock_instance.close.assert_called()
    
    @patch('pymongo.MongoClient')
    def test_setup_mongodb_database_connection_failure(self, mock_client):
        """Test MongoDB database setup with connection failure"""
        from setup_mongodb_deployment import setup_mongodb_database
        
        # Mock connection failure
        mock_client.side_effect = Exception("Connection failed")
        
        result = setup_mongodb_database()
        
        self.assertFalse(result)
    
    def test_setup_mongodb_database_no_pymongo(self):
        """Test MongoDB setup without pymongo installed"""
        from setup_mongodb_deployment import setup_mongodb_database
        
        # Mock ImportError for pymongo
        with patch('builtins.__import__', side_effect=ImportError("No module named 'pymongo'")):
            result = setup_mongodb_database()
            self.assertFalse(result)
    
    @patch('subprocess.run')
    @patch('builtins.input')
    def test_install_mlflow_mongodb_package_pypi(self, mock_input, mock_run):
        """Test installing package from PyPI"""
        from setup_mongodb_deployment import install_mlflow_mongodb_package
        
        # Mock user choice for PyPI
        mock_input.return_value = "1"
        
        # Mock successful installation
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            result = install_mlflow_mongodb_package()
            
            self.assertTrue(result)
            mock_run.assert_called()
    
    @patch('subprocess.run')
    @patch('builtins.input')
    def test_install_mlflow_mongodb_package_test_pypi(self, mock_input, mock_run):
        """Test installing package from Test PyPI"""
        from setup_mongodb_deployment import install_mlflow_mongodb_package
        
        # Mock user choice for Test PyPI
        mock_input.return_value = "2"
        
        # Mock successful installation
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            result = install_mlflow_mongodb_package()
            
            self.assertTrue(result)
    
    @patch('subprocess.run')
    @patch('builtins.input')
    def test_install_mlflow_mongodb_package_local(self, mock_input, mock_run):
        """Test installing package from local source"""
        from setup_mongodb_deployment import install_mlflow_mongodb_package
        
        # Create setup.py for local installation
        Path("setup.py").write_text("from setuptools import setup; setup(name='test')")
        
        # Mock user choice for local source
        mock_input.return_value = "3"
        
        # Mock successful installation
        mock_run.return_value = MagicMock(returncode=0)
        
        # Mock successful import
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            result = install_mlflow_mongodb_package()
            
            self.assertTrue(result)
    
    @patch('subprocess.run')
    @patch('builtins.input')
    def test_install_mlflow_mongodb_package_failure(self, mock_input, mock_run):
        """Test package installation failure"""
        from setup_mongodb_deployment import install_mlflow_mongodb_package
        
        # Mock user choice
        mock_input.return_value = "1"
        
        # Mock installation failure
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip install")
        
        result = install_mlflow_mongodb_package()
        
        self.assertFalse(result)
    
    @patch('builtins.input')
    def test_install_mlflow_mongodb_package_invalid_choice(self, mock_input):
        """Test package installation with invalid choice"""
        from setup_mongodb_deployment import install_mlflow_mongodb_package
        
        # Mock invalid choice
        mock_input.return_value = "invalid"
        
        result = install_mlflow_mongodb_package()
        
        self.assertFalse(result)
    
    def test_create_server_config(self):
        """Test creating server configuration"""
        from setup_mongodb_deployment import create_server_config
        
        result = create_server_config()
        
        self.assertTrue(result)
        self.assertTrue(Path("mlflow_config.json").exists())
        
        # Verify config content
        with open("mlflow_config.json", "r") as f:
            config = json.load(f)
            self.assertIn("mongodb", config)
            self.assertIn("server", config)
            self.assertEqual(config["server"]["port"], 5001)
    
    @patch('setup_mongodb_deployment.MongoDbTrackingStore')
    def test_test_installation_success(self, mock_store_class):
        """Test successful installation testing"""
        from setup_mongodb_deployment import test_installation
        
        # Mock successful imports
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            
            # Mock store operations
            mock_store = MagicMock()
            mock_store.search_experiments.return_value = []
            mock_store_class.return_value = mock_store
            
            result = test_installation()
            
            self.assertTrue(result)
    
    def test_test_installation_import_failure(self):
        """Test installation testing with import failure"""
        from setup_mongodb_deployment import test_installation
        
        # Mock import failure
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = test_installation()
            self.assertFalse(result)
    
    @patch('builtins.input')
    @patch('subprocess.run')
    def test_start_mlflow_server_create_script(self, mock_run, mock_input):
        """Test starting MLflow server with script creation"""
        from setup_mongodb_deployment import start_mlflow_server
        
        # Mock user choice to not start server
        mock_input.return_value = "n"
        
        result = start_mlflow_server()
        
        self.assertTrue(result)
        self.assertTrue(Path("mlflow_server_mongodb.py").exists())
        
        # Verify script content
        with open("mlflow_server_mongodb.py", "r") as f:
            content = f.read()
            self.assertIn("mongodb://admin:password@localhost:27017/mlflow", content)
    
    @patch('builtins.input')
    @patch('subprocess.run')
    def test_start_mlflow_server_start_now(self, mock_run, mock_input):
        """Test starting MLflow server immediately"""
        from setup_mongodb_deployment import start_mlflow_server
        
        # Create existing server script
        Path("mlflow_server_mongodb.py").write_text("#!/usr/bin/env python3\nprint('server')")
        
        # Mock user choice to start server
        mock_input.return_value = "y"
        
        # Mock server run (will be interrupted)
        mock_run.side_effect = KeyboardInterrupt()
        
        result = start_mlflow_server()
        
        self.assertTrue(result)
        mock_run.assert_called()
    
    def test_create_usage_guide(self):
        """Test creating usage guide"""
        from setup_mongodb_deployment import create_usage_guide
        
        result = create_usage_guide()
        
        self.assertTrue(result)
        self.assertTrue(Path("USAGE_GUIDE.md").exists())
        
        # Verify guide content
        with open("USAGE_GUIDE.md", "r") as f:
            content = f.read()
            self.assertIn("MLflow MongoDB Deployment", content)
            self.assertIn("http://localhost:5001", content)
            self.assertIn("Troubleshooting", content)


class TestSetupMongoDBIntegration(unittest.TestCase):
    """Integration tests for setup_mongodb_deployment.py"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up integration test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    @patch('setup_mongodb_deployment.check_system_requirements')
    @patch('setup_mongodb_deployment.setup_mongodb_database')
    @patch('setup_mongodb_deployment.install_mlflow_mongodb_package')
    @patch('setup_mongodb_deployment.test_installation')
    @patch('builtins.input')
    def test_main_workflow_success(self, mock_input, mock_test, mock_install, 
                                 mock_setup_db, mock_check_sys):
        """Test complete successful deployment workflow"""
        from setup_mongodb_deployment import main
        
        # Mock all operations to succeed
        mock_check_sys.return_value = True
        mock_setup_db.return_value = True
        mock_install.return_value = True
        mock_test.return_value = True
        
        # Mock user input to not start server
        mock_input.return_value = "n"
        
        result = main()
        
        self.assertTrue(result)
        
        # Verify files were created
        self.assertTrue(Path("mlflow_config.json").exists())
        self.assertTrue(Path("USAGE_GUIDE.md").exists())
        self.assertTrue(Path("mlflow_server_mongodb.py").exists())
    
    @patch('setup_mongodb_deployment.check_system_requirements')
    def test_main_workflow_system_requirements_failure(self, mock_check_sys):
        """Test workflow with system requirements failure"""
        from setup_mongodb_deployment import main
        
        # Mock system requirements failure
        mock_check_sys.return_value = False
        
        result = main()
        
        self.assertFalse(result)
    
    @patch('setup_mongodb_deployment.check_system_requirements')
    @patch('setup_mongodb_deployment.setup_mongodb_database')
    def test_main_workflow_database_setup_failure(self, mock_setup_db, mock_check_sys):
        """Test workflow with database setup failure"""
        from setup_mongodb_deployment import main
        
        # Mock system requirements success but database setup failure
        mock_check_sys.return_value = True
        mock_setup_db.return_value = False
        
        result = main()
        
        self.assertFalse(result)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
