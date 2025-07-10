"""
MongoDB Tracking Store Implementation for MLflow 3.0+

This module provides a MongoDB-based implementation of the MLflow tracking store.
"""

import json
import logging
import time
import uuid
from typing import Optional, List, Dict, Any

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from mlflow.entities import (
    Experiment,
    Run,
    RunInfo,
    RunData,
    RunStatus,
    ViewType,
    LifecycleStage,
    RunTag,
    Metric,
    Param,
    ExperimentTag,
    TraceInfoV2,
    DatasetInput,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
)
from mlflow.entities.metric import MetricWithRunId
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_experiment_name,
    _validate_run_id,
    _validate_tag_name,
    _validate_metric_name,
    _validate_param_name,
)

from mlflow_mongodb.db_utils import MongoDbUtils

_logger = logging.getLogger(__name__)


class MongoDbTrackingStore(AbstractStore):
    """
    MongoDB-based implementation of MLflow tracking store.
    
    This store uses MongoDB to persist experiment and run metadata.
    """
    
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        """
        Initialize MongoDB tracking store.
        
        Args:
            store_uri: MongoDB connection URI (e.g., mongodb://localhost:27017/mlflow)
            artifact_uri: Base URI for artifact storage (optional)
        """
        super().__init__()
        
        self.store_uri = store_uri
        self.artifact_uri = artifact_uri
        
        # Parse MongoDB URI
        self.connection_string, self.database_name, self.collection_prefix = (
            MongoDbUtils.parse_mongodb_uri(store_uri)
        )
        
        # Initialize MongoDB client and database
        self.client = MongoDbUtils.create_client(self.connection_string, self.database_name)
        self.db = self.client[self.database_name]
        
        # Initialize collections
        self.experiments_collection = self.db[f"{self.collection_prefix}_experiments"]
        self.runs_collection = self.db[f"{self.collection_prefix}_runs"]
        self.metrics_collection = self.db[f"{self.collection_prefix}_metrics"]
        self.params_collection = self.db[f"{self.collection_prefix}_params"]
        self.tags_collection = self.db[f"{self.collection_prefix}_tags"]
        self.latest_metrics_collection = self.db[f"{self.collection_prefix}_latest_metrics"]
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for optimal query performance"""
        MongoDbUtils.create_indexes_for_experiments(self.experiments_collection)
        MongoDbUtils.create_indexes_for_runs(self.runs_collection)
        MongoDbUtils.create_indexes_for_metrics(self.metrics_collection)
        MongoDbUtils.create_indexes_for_params(self.params_collection)
        MongoDbUtils.create_indexes_for_tags(self.tags_collection)
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        return str(int(time.time() * 1000000))
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        return str(uuid.uuid4().hex)
    
    def _experiment_doc_to_entity(self, doc: Dict[str, Any]) -> Experiment:
        """Convert MongoDB document to Experiment entity"""
        # Convert dict tags back to ExperimentTag objects
        tags = []
        for tag_dict in doc.get("tags", []):
            if isinstance(tag_dict, dict):
                tags.append(ExperimentTag(key=tag_dict["key"], value=tag_dict["value"]))
            else:
                tags.append(tag_dict)
        
        return Experiment(
            experiment_id=doc["experiment_id"],
            name=doc["name"],
            artifact_location=doc.get("artifact_location"),
            lifecycle_stage=doc.get("lifecycle_stage", LifecycleStage.ACTIVE),
            creation_time=doc.get("creation_time"),
            last_update_time=doc.get("last_update_time"),
            tags=tags,
        )
    
    def _run_doc_to_entity(self, doc: Dict[str, Any]) -> Run:
        """Convert MongoDB document to Run entity"""
        run_info = RunInfo(
            run_id=doc["run_uuid"],
            experiment_id=doc["experiment_id"],
            user_id=doc.get("user_id"),
            status=RunStatus.from_string(doc.get("status", "RUNNING")),
            start_time=doc.get("start_time"),
            end_time=doc.get("end_time"),
            lifecycle_stage=doc.get("lifecycle_stage", LifecycleStage.ACTIVE),
            artifact_uri=doc.get("artifact_uri"),
            run_name=doc.get("name"),
        )
        
        # Get run data (params, metrics, tags)
        run_data = self._get_run_data(doc["run_uuid"])
        
        return Run(run_info, run_data)
    
    def _get_run_data(self, run_id: str) -> RunData:
        """Get run data (params, metrics, tags) for a run"""
        # Get params
        params = []
        for param_doc in self.params_collection.find({"run_uuid": run_id}):
            params.append(Param(param_doc["key"], param_doc["value"]))
        
        # Get latest metrics
        metrics = []
        for metric_doc in self.latest_metrics_collection.find({"run_uuid": run_id}):
            metrics.append(Metric(
                metric_doc["key"],
                metric_doc["value"],
                metric_doc["timestamp"],
                metric_doc["step"]
            ))
        
        # Get tags
        tags = []
        for tag_doc in self.tags_collection.find({"run_uuid": run_id}):
            tags.append(RunTag(tag_doc["key"], tag_doc["value"]))
        
        return RunData(metrics, params, tags)
    
    # Experiment methods
    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ) -> PagedList[Experiment]:
        """Search for experiments"""
        query = {}
        
        # Apply view type filter
        if view_type == ViewType.ACTIVE_ONLY:
            query["lifecycle_stage"] = LifecycleStage.ACTIVE
        elif view_type == ViewType.DELETED_ONLY:
            query["lifecycle_stage"] = LifecycleStage.DELETED
        
        # Apply filter string (simplified implementation)
        if filter_string:
            # This would need more sophisticated parsing for full filter support
            pass
        
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
            sort_criteria = [("last_update_time", pymongo.DESCENDING)]
        
        # Execute query
        cursor = self.experiments_collection.find(query).sort(sort_criteria)
        
        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))
        
        experiments = []
        for doc in cursor.limit(max_results):
            experiments.append(self._experiment_doc_to_entity(doc))
        
        # Check if there are more results
        next_page_token = None
        if len(experiments) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)
        
        return PagedList(experiments, next_page_token)
    
    def create_experiment(self, name: str, artifact_location: str, tags: List[RunTag]) -> str:
        """Create a new experiment"""
        _validate_experiment_name(name)
        
        # Check if experiment already exists
        if self.experiments_collection.find_one({"name": name}):
            raise MlflowException(
                f"Experiment '{name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
        
        experiment_id = self._generate_experiment_id()
        current_time = get_current_time_millis()
        
        experiment_doc = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location or self.artifact_uri,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "creation_time": current_time,
            "last_update_time": current_time,
            "tags": [{"key": tag.key, "value": tag.value} for tag in (tags or [])],
        }
        
        try:
            self.experiments_collection.insert_one(experiment_doc)
            return experiment_id
        except pymongo.errors.DuplicateKeyError:
            raise MlflowException(
                f"Experiment '{name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
    
    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID"""
        doc = self.experiments_collection.find_one({"experiment_id": experiment_id})
        if not doc:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        return self._experiment_doc_to_entity(doc)
    
    def get_experiment_by_name(self, experiment_name: str) -> Experiment:
        """Get experiment by name"""
        doc = self.experiments_collection.find_one({"name": experiment_name})
        if not doc:
            raise MlflowException(
                f"Experiment '{experiment_name}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        return self._experiment_doc_to_entity(doc)
    
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete experiment"""
        result = self.experiments_collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "lifecycle_stage": LifecycleStage.DELETED,
                    "last_update_time": get_current_time_millis()
                }
            }
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def restore_experiment(self, experiment_id: str) -> None:
        """Restore deleted experiment"""
        result = self.experiments_collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "lifecycle_stage": LifecycleStage.ACTIVE,
                    "last_update_time": get_current_time_millis()
                }
            }
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        """Rename experiment"""
        _validate_experiment_name(new_name)
        
        # Check if new name already exists
        if self.experiments_collection.find_one({"name": new_name}):
            raise MlflowException(
                f"Experiment '{new_name}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
        
        result = self.experiments_collection.update_one(
            {"experiment_id": experiment_id},
            {
                "$set": {
                    "name": new_name,
                    "last_update_time": get_current_time_millis()
                }
            }
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
    
    # Run methods
    def create_run(self, experiment_id: str, user_id: str, start_time: int, tags: List[RunTag], run_name: str) -> Run:
        """Create a new run"""
        # Verify experiment exists
        if not self.experiments_collection.find_one({"experiment_id": experiment_id}):
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        run_id = self._generate_run_id()
        current_time = get_current_time_millis()
        
        run_doc = {
            "run_uuid": run_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": start_time or current_time,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "artifact_uri": f"{self.artifact_uri}/{experiment_id}/{run_id}/artifacts" if self.artifact_uri else None,
            "name": run_name,
        }
        
        try:
            self.runs_collection.insert_one(run_doc)
            
            # Insert tags
            if tags:
                tag_docs = [
                    {"run_uuid": run_id, "key": tag.key, "value": tag.value}
                    for tag in tags
                ]
                self.tags_collection.insert_many(tag_docs)
            
            return self._run_doc_to_entity(run_doc)
        except pymongo.errors.DuplicateKeyError:
            raise MlflowException(
                f"Run '{run_id}' already exists.",
                RESOURCE_ALREADY_EXISTS
            )
    
    def get_run(self, run_id: str) -> Run:
        """Get run by ID"""
        _validate_run_id(run_id)
        
        doc = self.runs_collection.find_one({"run_uuid": run_id})
        if not doc:
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        return self._run_doc_to_entity(doc)
    
    def update_run_info(self, run_id: str, run_status: RunStatus, end_time: int, run_name: str) -> RunInfo:
        """Update run info"""
        _validate_run_id(run_id)
        
        update_doc = {"last_update_time": get_current_time_millis()}
        
        if run_status:
            update_doc["status"] = RunStatus.to_string(run_status)
        if end_time:
            update_doc["end_time"] = end_time
        if run_name:
            update_doc["name"] = run_name
        
        result = self.runs_collection.update_one(
            {"run_uuid": run_id},
            {"$set": update_doc}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
        
        # Return updated run info
        return self.get_run(run_id).info
    
    def delete_run(self, run_id: str) -> None:
        """Delete run"""
        _validate_run_id(run_id)
        
        result = self.runs_collection.update_one(
            {"run_uuid": run_id},
            {"$set": {"lifecycle_stage": LifecycleStage.DELETED}}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def restore_run(self, run_id: str) -> None:
        """Restore deleted run"""
        _validate_run_id(run_id)
        
        result = self.runs_collection.update_one(
            {"run_uuid": run_id},
            {"$set": {"lifecycle_stage": LifecycleStage.ACTIVE}}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
    
    def log_metric(self, run_id: str, metric: Metric) -> None:
        """Log a metric"""
        _validate_run_id(run_id)
        _validate_metric_name(metric.key)
        
        # Insert metric
        metric_doc = {
            "run_uuid": run_id,
            "key": metric.key,
            "value": metric.value,
            "timestamp": metric.timestamp,
            "step": metric.step,
        }
        self.metrics_collection.insert_one(metric_doc)
        
        # Update latest metric
        self.latest_metrics_collection.replace_one(
            {"run_uuid": run_id, "key": metric.key},
            metric_doc,
            upsert=True
        )
    
    def log_param(self, run_id: str, param: Param) -> None:
        """Log a parameter"""
        _validate_run_id(run_id)
        _validate_param_name(param.key)
        
        param_doc = {
            "run_uuid": run_id,
            "key": param.key,
            "value": param.value,
        }
        
        try:
            self.params_collection.insert_one(param_doc)
        except pymongo.errors.DuplicateKeyError:
            # Parameter already exists, update it
            self.params_collection.update_one(
                {"run_uuid": run_id, "key": param.key},
                {"$set": {"value": param.value}}
            )
    
    def set_tag(self, run_id: str, tag: RunTag) -> None:
        """Set a tag"""
        _validate_run_id(run_id)
        _validate_tag_name(tag.key)
        
        tag_doc = {
            "run_uuid": run_id,
            "key": tag.key,
            "value": tag.value,
        }
        
        self.tags_collection.replace_one(
            {"run_uuid": run_id, "key": tag.key},
            tag_doc,
            upsert=True
        )
    
    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        """Get metric history"""
        _validate_run_id(run_id)
        
        metrics = []
        for doc in self.metrics_collection.find(
            {"run_uuid": run_id, "key": metric_key}
        ).sort("timestamp", pymongo.ASCENDING):
            metrics.append(Metric(
                doc["key"],
                doc["value"],
                doc["timestamp"],
                doc["step"]
            ))
        
        return metrics
    
    # Search methods (simplified implementations)
    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str = "",
        run_view_type: ViewType = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: List[str] = None,
        page_token: str = None,
    ) -> PagedList[Run]:
        """Search for runs"""
        # Convert RepeatedScalarContainer to list if needed
        if hasattr(experiment_ids, '__iter__') and not isinstance(experiment_ids, str):
            exp_ids = list(experiment_ids)
        else:
            exp_ids = experiment_ids if isinstance(experiment_ids, list) else [experiment_ids]
            
        query = {"experiment_id": {"$in": exp_ids}}
        
        # Apply view type filter
        if run_view_type == ViewType.ACTIVE_ONLY:
            query["lifecycle_stage"] = LifecycleStage.ACTIVE
        elif run_view_type == ViewType.DELETED_ONLY:
            query["lifecycle_stage"] = LifecycleStage.DELETED
        
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
            sort_criteria = [("start_time", pymongo.DESCENDING)]
        
        # Execute query
        cursor = self.runs_collection.find(query).sort(sort_criteria)
        
        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))
        
        runs = []
        for doc in cursor.limit(max_results):
            runs.append(self._run_doc_to_entity(doc))
        
        # Check if there are more results
        next_page_token = None
        if len(runs) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)
        
        return PagedList(runs, next_page_token)
    
    # Stub implementations for methods that would need full implementation
    def log_batch(self, run_id: str, metrics: List[Metric], params: List[Param], tags: List[RunTag]) -> None:
        """Log multiple metrics, params, and tags"""
        for metric in metrics:
            self.log_metric(run_id, metric)
        for param in params:
            self.log_param(run_id, param)
        for tag in tags:
            self.set_tag(run_id, tag)
    
    def record_logged_model(self, run_id: str, mlflow_model: Dict[str, Any]) -> None:
        """Record logged model (stub implementation)"""
        pass
    
    def search_logged_models(self, experiment_ids: List[str] = None, 
                           filter_string: str = None, max_results: int = 100,
                           order_by: List[str] = None, page_token: str = None) -> PagedList[LoggedModel]:
        """Search logged models (stub implementation)"""
        return PagedList([], None)
    
    def log_inputs(self, run_id: str, datasets: List[DatasetInput]) -> None:
        """Log input datasets (stub implementation)"""
        pass
    
    def search_traces(self, experiment_ids: List[str], filter_string: str = "", 
                     max_results: int = SEARCH_MAX_RESULTS_DEFAULT, 
                     order_by: List[str] = None, page_token: str = None) -> tuple:
        """Search traces (stub implementation)"""
        return PagedList([], None), None
    
    def start_trace(self, experiment_id: str, timestamp_ms: int, request_metadata: Dict[str, Any], 
                   tags: Dict[str, str]) -> TraceInfoV2:
        """Start trace (stub implementation)"""
        pass
    
    def end_trace(self, request_id: str, timestamp_ms: int, status: TraceStatus, 
                 request_metadata: Dict[str, Any], tags: Dict[str, str]) -> TraceInfoV2:
        """End trace (stub implementation)"""
        pass
    
    def get_trace_info(self, request_id: str) -> TraceInfoV2:
        """Get trace info (stub implementation)"""
        pass
    
    def set_trace_tag(self, request_id: str, key: str, value: str) -> None:
        """Set trace tag (stub implementation)"""
        pass
    
    def delete_trace_tag(self, request_id: str, key: str) -> None:
        """Delete trace tag (stub implementation)"""
        pass
