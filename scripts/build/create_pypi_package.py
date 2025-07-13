#!/usr/bin/env python3
"""
Script to create PyPI package for MLflow MongoDB integration
"""

import subprocess
import sys
import os
import shutil
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are installed"""
    logger.info("🔍 Checking prerequisites...")
    
    required_packages = ['build', 'twine', 'wheel']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, 
                         check=True, capture_output=True)
            logger.info("✓ All prerequisites installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install prerequisites: {e}")
            return False
    
    return True

def verify_package_structure():
    """Verify the package structure is correct"""
    logger.info("🔍 Verifying package structure...")
    
    # Required files for PyPI package (as per PYPI_PACKAGE_GUIDE.md)
    required_files = [
        "setup.py",
        "README.md",
        "requirements.txt",
        "mlflow_mongodb/__init__.py",
        "mlflow_mongodb/db_utils.py",
        "mlflow_mongodb/registration.py",
        "mlflow_mongodb/tracking/__init__.py",
        "mlflow_mongodb/tracking/mongodb_store.py",
        "mlflow_mongodb/model_registry/__init__.py",
        "mlflow_mongodb/model_registry/mongodb_store.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"✗ Missing required files: {missing_files}")
        return False
    
    logger.info(f"✓ All {len(required_files)} required files present")
    return True

def update_version():
    """Update version in setup.py"""
    logger.info("📝 Updating version...")
    
    try:
        # Read current setup.py
        with open("setup.py", "r") as f:
            content = f.read()
        
        # Extract current version
        import re
        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            current_version = version_match.group(1)
            logger.info(f"Current version: {current_version}")
            
            # Ask user if they want to update version
            new_version = input(f"Enter new version (current: {current_version}): ").strip()
            if new_version and new_version != current_version:
                # Update version in setup.py
                new_content = re.sub(
                    r'version\s*=\s*["\'][^"\']+["\']',
                    f'version="{new_version}"',
                    content
                )
                
                with open("setup.py", "w") as f:
                    f.write(new_content)
                
                logger.info(f"✓ Updated version to: {new_version}")
                return new_version
            else:
                logger.info(f"✓ Keeping current version: {current_version}")
                return current_version
        else:
            logger.error("✗ Could not find version in setup.py")
            return None
            
    except Exception as e:
        logger.error(f"✗ Failed to update version: {e}")
        return None

def clean_build_directories():
    """Clean previous build directories"""
    logger.info("🧹 Cleaning build directories...")
    
    build_dirs = ["build", "dist", "*.egg-info"]
    
    for pattern in build_dirs:
        import glob
        for path in glob.glob(pattern):
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.is_dir():
                    shutil.rmtree(path_obj)
                    logger.info(f"✓ Removed directory: {path}")
                else:
                    path_obj.unlink()
                    logger.info(f"✓ Removed file: {path}")
    
    return True

def build_package():
    """Build the package"""
    logger.info("🔨 Building package...")
    
    try:
        # Build source distribution and wheel
        result = subprocess.run([
            sys.executable, "-m", "build"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✓ Package built successfully")
        logger.info(f"Build output: {result.stdout}")
        
        # List built files
        dist_dir = Path("dist")
        if dist_dir.exists():
            built_files = list(dist_dir.glob("*"))
            logger.info(f"Built files: {[f.name for f in built_files]}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Build failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_package():
    """Check the built package"""
    logger.info("🔍 Checking package...")
    
    try:
        # Check with twine
        result = subprocess.run([
            sys.executable, "-m", "twine", "check", "dist/*"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✓ Package check passed")
        logger.info(f"Check output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Package check failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def upload_to_test_pypi():
    """Upload to Test PyPI"""
    logger.info("📤 Uploading to Test PyPI...")
    
    upload = input("Upload to Test PyPI? (y/N): ").strip().lower()
    if upload != 'y':
        logger.info("Skipping Test PyPI upload")
        return True
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "twine", "upload", 
            "--repository", "testpypi", 
            "dist/*"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✓ Uploaded to Test PyPI successfully")
        logger.info(f"Upload output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Test PyPI upload failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def upload_to_pypi():
    """Upload to PyPI"""
    logger.info("📤 Uploading to PyPI...")
    
    upload = input("Upload to PyPI? (y/N): ").strip().lower()
    if upload != 'y':
        logger.info("Skipping PyPI upload")
        return True
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "twine", "upload", "dist/*"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✓ Uploaded to PyPI successfully")
        logger.info(f"Upload output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ PyPI upload failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def create_installation_instructions(version):
    """Create installation instructions"""
    logger.info("📝 Creating installation instructions...")
    
    instructions = f"""
# MLflow MongoDB Integration - Installation Instructions

## Package Information
- Package Name: mlflow-mongodb
- Version: {version}
- PyPI URL: https://pypi.org/project/mlflow-mongodb/

## Installation

### From PyPI (Recommended)
```bash
pip install mlflow-mongodb
```

### From Test PyPI
```bash
pip install --index-url https://test.pypi.org/simple/ mlflow-mongodb
```

### From Source
```bash
git clone <repository-url>
cd mlflow-mongodb
pip install -e .
```

## Quick Start

1. Install the package:
   ```bash
   pip install mlflow-mongodb
   ```

2. Set up MongoDB (see setup_mongodb.py script)

3. Start MLflow server:
   ```bash
   python mlflow_server_mongodb.py
   ```

4. Access MLflow UI at http://localhost:5001

## Requirements
- Python >= 3.8
- MLflow >= 2.0.0
- pymongo >= 4.0.0
- MongoDB >= 4.4

## Support
For issues and questions, please visit the project repository.
"""
    
    with open("INSTALLATION.md", "w") as f:
        f.write(instructions)
    
    logger.info("✓ Created INSTALLATION.md")
    return True

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("  MLflow MongoDB PyPI Package Creator")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Verify package structure
    if not verify_package_structure():
        return False
    
    # Update version
    version = update_version()
    if not version:
        return False
    
    # Clean build directories
    if not clean_build_directories():
        return False
    
    # Build package
    if not build_package():
        return False
    
    # Check package
    if not check_package():
        return False
    
    # Create installation instructions
    if not create_installation_instructions(version):
        return False
    
    # Upload to Test PyPI (optional)
    if not upload_to_test_pypi():
        return False
    
    # Upload to PyPI (optional)
    if not upload_to_pypi():
        return False
    
    # Summary
    logger.info("=" * 60)
    logger.info("✅ PACKAGE CREATION COMPLETED!")
    logger.info(f"✅ Package version: {version}")
    logger.info("✅ Built files available in dist/")
    logger.info("✅ Installation instructions in INSTALLATION.md")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
