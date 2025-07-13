#!/bin/bash

# MLflow MongoDB Package Builder and Installer
# This script builds a wheel (.whl) file and installs it

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_DIR="mlflow_mongodb"
PACKAGE_NAME="mlflow-mongodb"
VENV_NAME=".newvenv"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate virtual environment
activate_venv() {
    if [ -d "$VENV_NAME" ]; then
        print_status "Activating virtual environment: $VENV_NAME"
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment $VENV_NAME not found"
        print_status "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment created and activated"
    fi
}

# Function to install build dependencies
install_build_deps() {
    print_status "Installing build dependencies..."
    pip install --upgrade pip
    pip install build wheel twine
    print_success "Build dependencies installed"
}

# Function to clean previous builds
clean_build() {
    print_status "Cleaning previous build artifacts..."
    
    cd "$PACKAGE_DIR"
    
    # Remove build directories
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Build artifacts cleaned"
    cd ..
}

# Function to build the wheel
build_wheel() {
    print_status "Building wheel package..."
    
    cd "$PACKAGE_DIR"
    
    # Build the package
    python -m build --wheel
    
    if [ $? -eq 0 ]; then
        print_success "Wheel built successfully"
        
        # List built files
        if [ -d "dist" ]; then
            print_status "Built files:"
            ls -la dist/
        fi
    else
        print_error "Failed to build wheel"
        cd ..
        exit 1
    fi
    
    cd ..
}

# Function to install the wheel
install_wheel() {
    print_status "Installing wheel package..."
    
    # Find the wheel file
    WHEEL_FILE=$(find "$PACKAGE_DIR/dist" -name "*.whl" | head -1)
    
    if [ -z "$WHEEL_FILE" ]; then
        print_error "No wheel file found in $PACKAGE_DIR/dist/"
        exit 1
    fi
    
    print_status "Installing: $WHEEL_FILE"
    
    # Uninstall existing package if it exists
    pip uninstall -y "$PACKAGE_NAME" 2>/dev/null || true
    
    # Install the wheel
    pip install "$WHEEL_FILE"
    
    if [ $? -eq 0 ]; then
        print_success "Package installed successfully"
    else
        print_error "Failed to install package"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Check if package can be imported
    python -c "import mlflow_mongodb; print('✓ mlflow_mongodb imported successfully')"

    if [ $? -eq 0 ]; then
        print_success "Installation verified"

        # Show package info
        print_status "Package information:"
        pip show "$PACKAGE_NAME"

        # Show package source location
        print_status "Package source location:"
        python -c "
import mlflow_mongodb
import os
from pathlib import Path

package_file = mlflow_mongodb.__file__
package_dir = Path(package_file).parent
cwd = Path.cwd()

print(f'Package file: {package_file}')

if str(package_dir).startswith(str(cwd)):
    print('📁 Source: PROJECT DIRECTORY (development mode)')
    print(f'   Relative path: {package_dir.relative_to(cwd)}')
elif 'site-packages' in str(package_dir):
    print('📦 Source: VIRTUAL ENVIRONMENT (installed)')
    print(f'   Site-packages path: {package_dir}')
else:
    print('❓ Source: OTHER LOCATION')
    print(f'   Path: {package_dir}')
"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Function to check package source
check_package_source() {
    print_status "Checking package source..."

    # Run the package source checker
    if [ -f "check_package_source.py" ]; then
        python check_package_source.py
    else
        # Simple inline check
        python -c "
import sys
import os
from pathlib import Path

print('Python executable:', sys.executable)
print('Virtual environment:', os.environ.get('VIRTUAL_ENV', 'None'))

try:
    import mlflow_mongodb
    package_file = mlflow_mongodb.__file__
    package_dir = Path(package_file).parent
    cwd = Path.cwd()

    print(f'\\nmlflow_mongodb location: {package_file}')

    if str(package_dir).startswith(str(cwd)):
        print('📁 Reading from: PROJECT DIRECTORY')
    elif 'site-packages' in str(package_dir):
        print('📦 Reading from: VIRTUAL ENVIRONMENT')
    else:
        print('❓ Reading from: OTHER LOCATION')

except ImportError:
    print('❌ mlflow_mongodb not found')
"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --clean    Clean build artifacts only"
    echo "  -b, --build    Build wheel only"
    echo "  -i, --install  Install existing wheel only"
    echo "  -v, --verify   Verify installation only"
    echo "  -s, --source   Check package source location"
    echo "  --no-venv      Skip virtual environment activation"
    echo ""
    echo "Default: Clean, build, and install the wheel package"
}

# Main function
main() {
    echo "=================================================="
    echo "  MLflow MongoDB Package Builder and Installer"
    echo "=================================================="
    
    # Parse command line arguments
    CLEAN_ONLY=false
    BUILD_ONLY=false
    INSTALL_ONLY=false
    VERIFY_ONLY=false
    SOURCE_ONLY=false
    USE_VENV=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--clean)
                CLEAN_ONLY=true
                shift
                ;;
            -b|--build)
                BUILD_ONLY=true
                shift
                ;;
            -i|--install)
                INSTALL_ONLY=true
                shift
                ;;
            -v|--verify)
                VERIFY_ONLY=true
                shift
                ;;
            -s|--source)
                SOURCE_ONLY=true
                shift
                ;;
            --no-venv)
                USE_VENV=false
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check if package directory exists
    if [ ! -d "$PACKAGE_DIR" ]; then
        print_error "Package directory '$PACKAGE_DIR' not found"
        exit 1
    fi
    
    # Check if Python is available
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Activate virtual environment if requested
    if [ "$USE_VENV" = true ]; then
        activate_venv
        install_build_deps
    fi
    
    # Execute based on options
    if [ "$CLEAN_ONLY" = true ]; then
        clean_build
    elif [ "$BUILD_ONLY" = true ]; then
        clean_build
        build_wheel
    elif [ "$INSTALL_ONLY" = true ]; then
        install_wheel
        verify_installation
    elif [ "$VERIFY_ONLY" = true ]; then
        verify_installation
    elif [ "$SOURCE_ONLY" = true ]; then
        check_package_source
    else
        # Default: full process
        clean_build
        build_wheel
        install_wheel
        verify_installation
    fi
    
    print_success "Process completed successfully!"
    
    # Show next steps
    echo ""
    echo "=================================================="
    echo "  Next Steps"
    echo "=================================================="
    echo "• Package is ready for use"
    echo "• Test with: python -c 'import mlflow_mongodb'"
    echo "• Start MLflow server with MongoDB backend"
    echo "• Upload to PyPI: twine upload $PACKAGE_DIR/dist/*"
    echo "=================================================="
}

# Run main function with all arguments
main "$@"
