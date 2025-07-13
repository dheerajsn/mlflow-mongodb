#!/usr/bin/env python3
"""
Simple PyPI package builder (non-interactive)
"""

import subprocess
import sys
import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # List built files
        dist_dir = Path("dist")
        if dist_dir.exists():
            built_files = list(dist_dir.glob("*"))
            logger.info(f"✓ Built files: {[f.name for f in built_files]}")
        
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
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Package check failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def verify_package_structure():
    """Verify the package structure is correct"""
    logger.info("🔍 Verifying package structure...")
    
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

def create_installation_instructions():
    """Create installation instructions"""
    logger.info("📝 Creating installation instructions...")
    
    # Get version from setup.py
    version = "1.0.1"  # Default version
    try:
        with open("setup.py", "r") as f:
            content = f.read()
            import re
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if version_match:
                version = version_match.group(1)
    except:
        pass
    
    instructions = f"""
# MLflow MongoDB Integration - Installation Instructions

## Package Information
- Package Name: mlflow-mongodb
- Version: {version}
- Built: {Path('dist').exists() and 'Available in dist/' or 'Not built yet'}

## Installation

### From Local Build
```bash
pip install dist/mlflow-mongodb-{version}.tar.gz
```

### From Wheel
```bash
pip install dist/mlflow_mongodb-{version}-py3-none-any.whl
```

### Development Installation
```bash
pip install -e .
```

## Quick Start

1. Install the package from local build
2. Set up MongoDB (see setup_mongodb_deployment.py script)
3. Start MLflow server with MongoDB backend

## Upload to PyPI (when ready)

### Test PyPI
```bash
python -m twine upload --repository testpypi dist/*
```

### Production PyPI
```bash
python -m twine upload dist/*
```

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
    logger.info("  MLflow MongoDB PyPI Package Builder")
    logger.info("=" * 60)
    
    # Verify package structure
    if not verify_package_structure():
        logger.error("❌ Package structure verification failed")
        return False
    
    # Clean build directories
    if not clean_build_directories():
        logger.error("❌ Failed to clean build directories")
        return False
    
    # Build package
    if not build_package():
        logger.error("❌ Package build failed")
        return False
    
    # Check package
    if not check_package():
        logger.error("❌ Package check failed")
        return False
    
    # Create installation instructions
    if not create_installation_instructions():
        logger.error("❌ Failed to create installation instructions")
        return False
    
    # Summary
    logger.info("=" * 60)
    logger.info("✅ PACKAGE BUILD COMPLETED!")
    logger.info("✅ Built files available in dist/")
    logger.info("✅ Installation instructions in INSTALLATION.md")
    logger.info("✅ Ready for local installation or PyPI upload")
    logger.info("=" * 60)
    
    # Show next steps
    logger.info("\n📋 Next Steps:")
    logger.info("1. Test locally: pip install dist/mlflow-mongodb-*.tar.gz")
    logger.info("2. Upload to Test PyPI: python -m twine upload --repository testpypi dist/*")
    logger.info("3. Upload to PyPI: python -m twine upload dist/*")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
