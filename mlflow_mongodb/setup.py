#!/usr/bin/env python3
"""
Setup script for MLflow MongoDB Plugin

This script sets up the MLflow MongoDB plugin with proper entry points
for tracking and model registry stores.
"""

from setuptools import setup, find_packages

setup(
    name="mlflow-mongodb",
    version="1.1.0",
    description="MongoDB backend for MLflow tracking and model registry",
    author="MongoDB MLflow Plugin",
    author_email="support@example.com",
    py_modules=["__init__", "db_utils", "registration"],
    packages=["tracking", "model_registry"],
    install_requires=[
        "mlflow>=3.0.0",
        "pymongo>=4.0.0",
        "urllib3>=1.26.0",
    ],
    python_requires=">=3.8",
    entry_points={
        # MLflow plugin entry points
        "mlflow.tracking_store": [
            "mongodb=mlflow_mongodb.tracking.mongodb_store:MongoDbTrackingStore"
        ],
        "mlflow.model_registry_store": [
            "mongodb=mlflow_mongodb.model_registry.mongodb_store:MongoDbModelRegistryStore"
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
