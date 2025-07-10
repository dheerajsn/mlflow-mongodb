"""
MongoDB Model Registry Store Implementation for MLflow 3.0+

This module provides a MongoDB-based implementation of the MLflow model registry store.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from mlflow.entities.model_registry import (
    RegisteredModel,
    ModelVersion,
    ModelVersionTag,
    RegisteredModelTag,
    RegisteredModelAlias,
)
from mlflow.entities.model_registry.model_version_stages import STAGE_NONE
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
)
from mlflow.utils.time import get_current_time_millis

from mlflow_mongodb.db_utils import MongoDbUtils

_logger = logging.getLogger(__name__)


class MongoDbModelRegistryStore(AbstractStore):
    """
    MongoDB-based implementation of MLflow model registry store.
    
    This store uses MongoDB to persist model registry metadata.
    """
    
    def __init__(self, store_uri: str, tracking_uri: Optional[str] = None):
        """
        Initialize MongoDB model registry store.
        
        Args:
            store_uri: MongoDB connection URI (e.g., mongodb://localhost:27017/mlflow)
            tracking_uri: URI of the tracking server (optional)
        """
        super().__init__(store_uri, tracking_uri)
        
        self.store_uri = store_uri
        self.tracking_uri = tracking_uri
        
        # Parse MongoDB URI
        self.connection_string, self.database_name, self.collection_prefix = (
            MongoDbUtils.parse_mongodb_uri(store_uri)
        )
        
        # Initialize MongoDB client and database
        self.client = MongoDbUtils.create_client(self.connection_string, self.database_name)
        self.db = self.client[self.database_name]
        
        # Initialize collections
        self.registered_models_collection = self.db[f"{self.collection_prefix}_registered_models"]
        self.model_versions_collection = self.db[f"{self.collection_prefix}_model_versions"]
        self.model_version_tags_collection = self.db[f"{self.collection_prefix}_model_version_tags"]
        self.registered_model_tags_collection = self.db[f"{self.collection_prefix}_registered_model_tags"]
        self.registered_model_aliases_collection = self.db[f"{self.collection_prefix}_registered_model_aliases"]
        
        # Prompts are stored as registered models with special tags, no separate collections needed
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for optimal query performance"""
        MongoDbUtils.create_indexes_for_registered_models(self.registered_models_collection)
        MongoDbUtils.create_indexes_for_model_versions(self.model_versions_collection)
        
        # Create additional indexes for tags to support prompt operations
        self.registered_model_tags_collection.create_index([("name", 1), ("key", 1)], unique=True)
        self.model_version_tags_collection.create_index([("name", 1), ("version", 1), ("key", 1)], unique=True)
    
    def _registered_model_doc_to_entity(self, doc: Dict[str, Any]) -> RegisteredModel:
        """Convert MongoDB document to RegisteredModel entity"""
        # Get latest versions for each stage
        latest_versions = []
        model_versions = list(self.model_versions_collection.find({"name": doc["name"]}))
        
        # Group by stage and find latest version for each
        stage_versions = {}
        for mv_doc in model_versions:
            stage = mv_doc.get("current_stage", STAGE_NONE)
            if stage not in stage_versions or stage_versions[stage]["version"] < mv_doc["version"]:
                stage_versions[stage] = mv_doc
        
        for mv_doc in stage_versions.values():
            latest_versions.append(self._model_version_doc_to_entity(mv_doc))
        
        # Get tags
        tags = []
        for tag_doc in self.registered_model_tags_collection.find({"name": doc["name"]}):
            tags.append(RegisteredModelTag(tag_doc["key"], tag_doc["value"]))
        
        # Get aliases
        aliases = []
        for alias_doc in self.registered_model_aliases_collection.find({"name": doc["name"]}):
            aliases.append(RegisteredModelAlias(alias_doc["alias"], alias_doc["version"]))
        
        return RegisteredModel(
            name=doc["name"],
            creation_timestamp=doc.get("creation_time"),
            last_updated_timestamp=doc.get("last_updated_time"),
            description=doc.get("description"),
            latest_versions=latest_versions,
            tags=tags,
            aliases=aliases,
        )
    
    def _model_version_doc_to_entity(self, doc: Dict[str, Any]) -> ModelVersion:
        """Convert MongoDB document to ModelVersion entity"""
        # Get tags as list of tag objects
        tags = []
        for tag_doc in self.model_version_tags_collection.find({
            "name": doc["name"],
            "version": doc["version"]
        }):
            tags.append(ModelVersionTag(tag_doc["key"], tag_doc["value"]))
        
        return ModelVersion(
            name=doc["name"],
            version=str(doc["version"]),
            creation_timestamp=doc.get("creation_time"),
            last_updated_timestamp=doc.get("last_updated_time"),
            description=doc.get("description"),
            user_id=doc.get("user_id"),
            current_stage=doc.get("current_stage", STAGE_NONE),
            source=doc.get("source"),
            run_id=doc.get("run_id"),
            run_link=doc.get("run_link"),
            status=ModelVersionStatus.to_string(ModelVersionStatus.from_string(doc.get("status", "READY"))),
            status_message=doc.get("status_message"),
            tags=tags,
        )
    
    def _prompt_doc_to_entity(self, doc: Dict[str, Any]) -> Prompt:
        """Convert MongoDB registered model document to Prompt entity"""
        # Note: Prompt entity does not include versions - those are handled separately
        
        # Get tags from registered_model_tags_collection
        tags = {}
        for tag_doc in self.registered_model_tags_collection.find({"name": doc["name"]}):
            # Skip internal MLflow prompt tags
            if not tag_doc["key"].startswith("mlflow.prompt."):
                tags[tag_doc["key"]] = tag_doc["value"]
        
        return Prompt(
            name=doc["name"],
            description=doc.get("description"),
            creation_timestamp=doc.get("creation_time"),
            tags=tags,
        )
    
    def _prompt_version_doc_to_entity(self, doc: Dict[str, Any]) -> PromptVersion:
        """Convert MongoDB model version document to PromptVersion entity"""
        # Get tags from model_version_tags_collection
        tags = {}
        prompt_text = ""
        
        for tag_doc in self.model_version_tags_collection.find({
            "name": doc["name"],
            "version": doc["version"]
        }):
            key = tag_doc["key"]
            value = tag_doc["value"]
            
            if key == "mlflow.prompt.text":
                prompt_text = value
            elif not key.startswith("mlflow.prompt."):
                # Only include user tags, not internal MLflow tags
                tags[key] = value
        
        return PromptVersion(
            name=doc["name"],
            version=int(doc["version"]),  # PromptVersion expects int, not str
            template=prompt_text,
            creation_timestamp=doc.get("creation_time"),
            last_updated_timestamp=doc.get("last_updated_time"),
            tags=tags,
        )
    
    def _get_next_version_number(self, name: str) -> int:
        """Get the next version number for a model using atomic increment"""
        # Use findAndModify to atomically increment version counter
        result = self.registered_models_collection.find_one_and_update(
            {"name": name},
            {"$inc": {"version_counter": 1}},
            return_document=pymongo.ReturnDocument.AFTER,
            upsert=False
        )
        
        if result and "version_counter" in result:
            return result["version_counter"]
        else:
            # Fallback: find highest existing version and add 1
            version_doc = self.model_versions_collection.find_one(
                {"name": name},
                sort=[("version", pymongo.DESCENDING)]
            )
            next_version = (version_doc["version"] + 1) if version_doc else 1
            
            # Initialize the counter in the registered model
            self.registered_models_collection.update_one(
                {"name": name},
                {"$set": {"version_counter": next_version}}
            )
            return next_version
    
    
    # Registered Model CRUD operations
    def create_registered_model(self, name: str, tags: List[RegisteredModelTag] = None, 
                               description: str = None, deployment_job_id: str = None) -> RegisteredModel:
        """Create a new registered model"""
        # Check if model already exists
        if self.registered_models_collection.find_one({"name": name}):
            raise MlflowException(
                f"Registered model '{name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
        
        current_time = get_current_time_millis()
        
        model_doc = {
            "name": name,
            "creation_time": current_time,
            "last_updated_time": current_time,
            "description": description,
            "deployment_job_id": deployment_job_id,
            "version_counter": 0,  # Initialize version counter
        }
        
        try:
            self.registered_models_collection.insert_one(model_doc)
            
            # Insert tags
            if tags:
                tag_docs = [
                    {"name": name, "key": tag.key, "value": tag.value}
                    for tag in tags
                ]
                self.registered_model_tags_collection.insert_many(tag_docs)
            
            return self._registered_model_doc_to_entity(model_doc)
        except pymongo.errors.DuplicateKeyError:
            raise MlflowException(
                f"Registered model '{name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
    
    def update_registered_model(self, name: str, description: str = None, 
                               deployment_job_id: str = None) -> RegisteredModel:
        """Update registered model description"""
        update_doc = {"last_updated_time": get_current_time_millis()}
        
        if description is not None:
            update_doc["description"] = description
        if deployment_job_id is not None:
            update_doc["deployment_job_id"] = deployment_job_id
        
        result = self.registered_models_collection.update_one(
            {"name": name},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return self.get_registered_model(name)
    
    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """Rename registered model"""
        # Check if new name already exists
        if self.registered_models_collection.find_one({"name": new_name}):
            raise MlflowException(
                f"Registered model '{new_name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
        
        # Update registered model
        result = self.registered_models_collection.update_one(
            {"name": name},
            {
                "$set": {
                    "name": new_name,
                    "last_updated_time": get_current_time_millis()
                }
            }
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Update all related collections
        self.model_versions_collection.update_many(
            {"name": name},
            {"$set": {"name": new_name}}
        )
        self.model_version_tags_collection.update_many(
            {"name": name},
            {"$set": {"name": new_name}}
        )
        self.registered_model_tags_collection.update_many(
            {"name": name},
            {"$set": {"name": new_name}}
        )
        self.registered_model_aliases_collection.update_many(
            {"name": name},
            {"$set": {"name": new_name}}
        )
        
        return self.get_registered_model(new_name)
    
    def delete_registered_model(self, name: str) -> None:
        """Delete registered model"""
        result = self.registered_models_collection.delete_one({"name": name})
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Delete related data
        self.model_versions_collection.delete_many({"name": name})
        self.model_version_tags_collection.delete_many({"name": name})
        self.registered_model_tags_collection.delete_many({"name": name})
        self.registered_model_aliases_collection.delete_many({"name": name})
    
    def search_registered_models(self, filter_string: str = None, 
                                max_results: int = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
                                order_by: List[str] = None, 
                                page_token: str = None) -> PagedList[RegisteredModel]:
        """Search registered models (excludes prompts by default, includes prompts only when specifically requested)"""
        from mlflow.utils.search_utils import SearchModelUtils, SearchUtils
        from mlflow.prompt.registry_utils import add_prompt_filter_string
        
        # Validation
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        
        # Apply prompt filter logic following the reference file store implementation
        # By default, exclude prompts from search results
        filter_string = add_prompt_filter_string(filter_string, is_prompt=False)
        
        # Get all registered models
        all_models = []
        for doc in self.registered_models_collection.find():
            all_models.append(self._registered_model_doc_to_entity(doc))
        
        # Apply filter using MLflow's standard utilities
        filtered_models = SearchModelUtils.filter(all_models, filter_string)
        
        # Apply sorting using MLflow's standard utilities
        sorted_models = SearchModelUtils.sort(filtered_models, order_by)
        
        # Apply pagination
        start_offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_models = sorted_models[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(sorted_models):
            next_page_token = SearchUtils.create_page_token(final_offset)

        return PagedList(paginated_models, next_page_token)
    
    def _is_querying_prompt(self, filter_string: str) -> bool:
        """Check if the filter is specifically requesting prompts"""
        if not filter_string:
            return False
        
        # Simple check for prompt tag filter
        # In a full implementation, we'd need proper filter parsing
        return "mlflow.prompt.is_prompt" in filter_string and "true" in filter_string.lower()
    
    def get_registered_model(self, name: str) -> RegisteredModel:
        """Get registered model by name"""
        doc = self.registered_models_collection.find_one({"name": name})
        if not doc:
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        return self._registered_model_doc_to_entity(doc)
    
    def get_latest_versions(self, name: str, stages: List[str] = None) -> List[ModelVersion]:
        """Get latest versions for specified stages"""
        query = {"name": name}
        
        if stages:
            query["current_stage"] = {"$in": stages}
        
        # Group by stage and find latest version for each
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": "$current_stage",
                "doc": {"$first": "$$ROOT"},
                "max_version": {"$max": "$version"}
            }},
            {"$replaceRoot": {"newRoot": "$doc"}}
        ]
        
        versions = []
        for doc in self.model_versions_collection.aggregate(pipeline):
            versions.append(self._model_version_doc_to_entity(doc))
        
        return versions
    
    # Model Version CRUD operations
    def create_model_version(self, name: str, source: str, run_id: str = None, 
                            tags: List[ModelVersionTag] = None, run_link: str = None,
                            description: str = None, local_model_path: str = None,
                            model_id: Optional[str] = None) -> ModelVersion:
        """Create a new model version"""
        # Check if registered model exists
        if not self.registered_models_collection.find_one({"name": name}):
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        version = self._get_next_version_number(name)
        current_time = get_current_time_millis()
        
        version_doc = {
            "name": name,
            "version": version,
            "creation_time": current_time,
            "last_updated_time": current_time,
            "description": description,
            "user_id": None,  # Would need to get from context
            "current_stage": STAGE_NONE,
            "source": source,
            "local_model_path": local_model_path,
            "model_id": model_id,
            "run_id": run_id,
            "run_link": run_link,
            "status": ModelVersionStatus.to_string(ModelVersionStatus.READY),
            "status_message": None,
        }
        
        try:
            self.model_versions_collection.insert_one(version_doc)
            
            # Insert tags
            if tags:
                tag_docs = [
                    {"name": name, "version": version, "key": tag.key, "value": tag.value}
                    for tag in tags
                ]
                self.model_version_tags_collection.insert_many(tag_docs)
            
            # Update registered model timestamp
            self.registered_models_collection.update_one(
                {"name": name},
                {"$set": {"last_updated_time": current_time}}
            )
            
            return self._model_version_doc_to_entity(version_doc)
        except pymongo.errors.DuplicateKeyError:
            raise MlflowException(
                f"Model version '{name}' version '{version}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
    
    def update_model_version(self, name: str, version: str, description: str = None) -> ModelVersion:
        """Update model version description"""
        update_doc = {"last_updated_time": get_current_time_millis()}
        
        if description is not None:
            update_doc["description"] = description
        
        result = self.model_versions_collection.update_one(
            {"name": name, "version": int(version)},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return self.get_model_version(name, version)
    
    def transition_model_version_stage(self, name: str, version: str, stage: str, 
                                      archive_existing_versions: bool = False) -> ModelVersion:
        """Transition model version to a new stage"""
        # Archive existing versions in the same stage if requested
        if archive_existing_versions:
            self.model_versions_collection.update_many(
                {"name": name, "current_stage": stage},
                {"$set": {"current_stage": "Archived", "last_updated_time": get_current_time_millis()}}
            )
        
        # Update the target version
        result = self.model_versions_collection.update_one(
            {"name": name, "version": int(version)},
            {
                "$set": {
                    "current_stage": stage,
                    "last_updated_time": get_current_time_millis()
                }
            }
        )
        
        if result.matched_count == 0:
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return self.get_model_version(name, version)
    
    def delete_model_version(self, name: str, version: str) -> None:
        """Delete model version"""
        result = self.model_versions_collection.delete_one({
            "name": name,
            "version": int(version)
        })
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Delete related tags
        self.model_version_tags_collection.delete_many({
            "name": name,
            "version": int(version)
        })
    
    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Get model version by name and version"""
        doc = self.model_versions_collection.find_one({
            "name": name,
            "version": int(version)
        })
        if not doc:
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        return self._model_version_doc_to_entity(doc)
    
    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """Get download URI for model version"""
        model_version = self.get_model_version(name, version)
        return model_version.source
    
    def search_model_versions(self, filter_string: str = None, 
                             max_results: int = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
                             order_by: List[str] = None, 
                             page_token: str = None) -> PagedList[ModelVersion]:
        """Search model versions"""
        from mlflow.utils.search_utils import SearchModelVersionUtils, SearchUtils
        from mlflow.prompt.registry_utils import add_prompt_filter_string
        
        # Validation (following reference implementations)
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        
        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        
        # Get all model versions from MongoDB
        all_versions = []
        for doc in self.model_versions_collection.find():
            all_versions.append(self._model_version_doc_to_entity(doc))
        
        # Apply prompt filter logic following the reference file store implementation
        # By default, exclude prompts from search results (like file store does)
        filter_string = add_prompt_filter_string(filter_string, is_prompt=False)
        
        # Apply filter using MLflow's standard utilities
        filtered_versions = SearchModelVersionUtils.filter(all_versions, filter_string)
        
        # Apply sorting using MLflow's standard utilities
        sorted_versions = SearchModelVersionUtils.sort(
            filtered_versions,
            order_by or ["last_updated_timestamp DESC", "name ASC", "version_number DESC"],
        )
        
        # Apply pagination
        start_offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results
        
        paginated_versions = sorted_versions[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(sorted_versions):
            next_page_token = SearchUtils.create_page_token(final_offset)
        
        return PagedList(paginated_versions, next_page_token)
    
    # Tag operations
    def set_registered_model_tag(self, name: str, tag: RegisteredModelTag) -> None:
        """Set tag on registered model"""
        # Check if model exists
        if not self.registered_models_collection.find_one({"name": name}):
            raise MlflowException(
                f"Registered model '{name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        tag_doc = {
            "name": name,
            "key": tag.key,
            "value": tag.value,
        }
        
        self.registered_model_tags_collection.replace_one(
            {"name": name, "key": tag.key},
            tag_doc,
            upsert=True
        )
    
    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete tag from registered model"""
        result = self.registered_model_tags_collection.delete_one({
            "name": name,
            "key": key
        })
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Registered model tag '{key}' not found for model '{name}'",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def set_model_version_tag(self, name: str, version: str, tag: ModelVersionTag) -> None:
        """Set tag on model version"""
        # Check if model version exists
        if not self.model_versions_collection.find_one({"name": name, "version": int(version)}):
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        tag_doc = {
            "name": name,
            "version": int(version),
            "key": tag.key,
            "value": tag.value,
        }
        
        self.model_version_tags_collection.replace_one(
            {"name": name, "version": int(version), "key": tag.key},
            tag_doc,
            upsert=True
        )
    
    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        """Delete tag from model version"""
        result = self.model_version_tags_collection.delete_one({
            "name": name,
            "version": int(version),
            "key": key
        })
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Model version tag '{key}' not found for model '{name}' version '{version}'",
                RESOURCE_DOES_NOT_EXIST
            )
    
    # Alias operations (simplified implementations)
    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        """Set alias for registered model version"""
        # Check if model version exists
        if not self.model_versions_collection.find_one({"name": name, "version": int(version)}):
            raise MlflowException(
                f"Model version '{name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        alias_doc = {
            "name": name,
            "alias": alias,
            "version": int(version),
        }
        
        self.registered_model_aliases_collection.replace_one(
            {"name": name, "alias": alias},
            alias_doc,
            upsert=True
        )
    
    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        """Delete alias from registered model"""
        result = self.registered_model_aliases_collection.delete_one({
            "name": name,
            "alias": alias
        })
        
        if result.deleted_count == 0:
            raise MlflowException(
                f"Registered model alias '{alias}' not found for model '{name}'",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        """Get model version by alias"""
        alias_doc = self.registered_model_aliases_collection.find_one({
            "name": name,
            "alias": alias
        })
        
        if not alias_doc:
            raise MlflowException(
                f"Registered model alias '{alias}' not found for model '{name}'",
                RESOURCE_DOES_NOT_EXIST
            )
        
        return self.get_model_version(name, str(alias_doc["version"]))
    
    def search_prompts(self, filter_string: str = None, 
                      max_results: int = 100, order_by: List[str] = None,
                      page_token: str = None) -> PagedList[Prompt]:
        """Search prompts (stored as registered models with prompt tags)"""
        # First, find all registered models that have the prompt marker tag
        prompt_names = set()
        for tag_doc in self.registered_model_tags_collection.find({"key": "mlflow.prompt.is_prompt"}):
            prompt_names.add(tag_doc["name"])
        
        if not prompt_names:
            return PagedList([], None)
        
        # Build query for registered models that are prompts
        query = {"name": {"$in": list(prompt_names)}}
        
        # Apply filter (simplified implementation)
        if filter_string:
            # Apply regex filter while preserving the $in constraint for prompt names
            query = {
                "name": {
                    "$regex": filter_string,
                    "$options": "i",
                    "$in": list(prompt_names)
                }
            }
        
        # Apply ordering
        sort_criteria = []
        if order_by:
            for order_item in order_by:
                if "DESC" in order_item:
                    field = order_item.replace(" DESC", "")
                    sort_criteria.append((field, pymongo.DESCENDING))
                else:
                    field = order_item.replace(" ASC", "")
                    sort_criteria.append((field, pymongo.ASCENDING))
        else:
            sort_criteria = [("last_updated_time", pymongo.DESCENDING)]
        
        # Execute query on registered_models_collection
        cursor = self.registered_models_collection.find(query).sort(sort_criteria)
        
        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))
        
        prompts = []
        for doc in cursor.limit(max_results):
            prompts.append(self._prompt_doc_to_entity(doc))
        
        # Check if there are more results
        next_page_token = None
        if len(prompts) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)
        
        return PagedList(prompts, next_page_token)
    
    def search_prompt_versions(self, prompt_name: str, filter_string: str = None, 
                              max_results: int = 100, order_by: List[str] = None,
                              page_token: str = None) -> PagedList[PromptVersion]:
        """Search prompt versions for a specific prompt"""
        # Check if the prompt exists
        if not self.registered_models_collection.find_one({"name": prompt_name}):
            raise MlflowException(
                f"Prompt '{prompt_name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Check if it's actually a prompt
        prompt_tag = self.registered_model_tags_collection.find_one({
            "name": prompt_name, 
            "key": "mlflow.prompt.is_prompt"
        })
        if not prompt_tag:
            raise MlflowException(
                f"'{prompt_name}' is not a prompt",
                INVALID_PARAMETER_VALUE
            )
        
        # Get all versions for this prompt
        query = {"name": prompt_name}
        
        # Apply ordering
        sort_criteria = []
        if order_by:
            for order_item in order_by:
                if "DESC" in order_item:
                    field = order_item.replace(" DESC", "").replace("version_number", "version")
                    sort_criteria.append((field, pymongo.DESCENDING))
                else:
                    field = order_item.replace(" ASC", "").replace("version_number", "version")
                    sort_criteria.append((field, pymongo.ASCENDING))
        else:
            sort_criteria = [("version", pymongo.DESCENDING)]
        
        # Execute query
        cursor = self.model_versions_collection.find(query).sort(sort_criteria)
        
        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))
        
        prompt_versions = []
        for doc in cursor.limit(max_results):
            prompt_versions.append(self._prompt_version_doc_to_entity(doc))
        
        # Check if there are more results
        next_page_token = None
        if len(prompt_versions) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)
        
        return PagedList(prompt_versions, next_page_token)
    
    def get_prompt(self, prompt_name: str) -> Prompt:
        """Get prompt by name"""
        doc = self.registered_models_collection.find_one({"name": prompt_name})
        if not doc:
            raise MlflowException(
                f"Prompt '{prompt_name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Check if it's actually a prompt
        prompt_tag = self.registered_model_tags_collection.find_one({
            "name": prompt_name, 
            "key": "mlflow.prompt.is_prompt"
        })
        if not prompt_tag:
            raise MlflowException(
                f"'{prompt_name}' is not a prompt",
                INVALID_PARAMETER_VALUE
            )
        
        return self._prompt_doc_to_entity(doc)
    
    def get_prompt_version(self, prompt_name: str, version: str) -> PromptVersion:
        """Get specific prompt version"""
        doc = self.model_versions_collection.find_one({
            "name": prompt_name,
            "version": int(version)
        })
        if not doc:
            raise MlflowException(
                f"Prompt version '{prompt_name}' version '{version}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Check if it's actually a prompt
        prompt_tag = self.registered_model_tags_collection.find_one({
            "name": prompt_name, 
            "key": "mlflow.prompt.is_prompt"
        })
        if not prompt_tag:
            raise MlflowException(
                f"'{prompt_name}' is not a prompt",
                INVALID_PARAMETER_VALUE
            )
        
        return self._prompt_version_doc_to_entity(doc)
