"""
MongoDB Tracking Store Implementation for MLflow 3.0+

This module provides a MongoDB-based implementation of the MLflow tracking store.
"""

import json
import logging
import time
import uuid
from typing import Optional, List, Dict, Any, Union

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
        try:
            self.client = MongoDbUtils.create_client(self.connection_string, self.database_name)
            self.db = self.client[self.database_name]

            # Test the connection
            self.client.admin.command('ping')
            _logger.info(f"Successfully connected to MongoDB: {self.database_name}")
        except Exception as e:
            _logger.error(f"Failed to connect to MongoDB: {e}")
            raise MlflowException(
                f"Failed to connect to MongoDB: {e}",
                INTERNAL_ERROR
            )
        
        # Initialize collections
        self.experiments_collection = self.db[f"{self.collection_prefix}_experiments"]
        self.runs_collection = self.db[f"{self.collection_prefix}_runs"]
        self.metrics_collection = self.db[f"{self.collection_prefix}_metrics"]
        self.params_collection = self.db[f"{self.collection_prefix}_params"]
        self.tags_collection = self.db[f"{self.collection_prefix}_tags"]
        self.latest_metrics_collection = self.db[f"{self.collection_prefix}_latest_metrics"]
        self.logged_models_collection = self.db[f"{self.collection_prefix}_logged_models"]
        self.dataset_inputs_collection = self.db[f"{self.collection_prefix}_dataset_inputs"]
        self.traces_collection = self.db[f"{self.collection_prefix}_traces"]
        self.trace_artifacts_collection = self.db[f"{self.collection_prefix}_trace_artifacts"]

        # Create indexes with error handling
        try:
            self._create_indexes()
            _logger.info("Successfully created MongoDB indexes")
        except Exception as e:
            _logger.warning(f"Failed to create some indexes: {e}")
            # Continue anyway - indexes are for performance, not functionality
    
    def _create_indexes(self):
        """Create necessary indexes for optimal query performance"""
        MongoDbUtils.create_indexes_for_experiments(self.experiments_collection)
        MongoDbUtils.create_indexes_for_runs(self.runs_collection)
        MongoDbUtils.create_indexes_for_metrics(self.metrics_collection)
        MongoDbUtils.create_indexes_for_params(self.params_collection)
        MongoDbUtils.create_indexes_for_tags(self.tags_collection)

        # Create additional indexes for new collections
        self._create_logged_models_indexes()
        self._create_dataset_inputs_indexes()
        self._create_traces_indexes()
        self._create_trace_artifacts_indexes()

    def _create_logged_models_indexes(self):
        """Create indexes for logged models collection"""
        indexes = [
            pymongo.IndexModel([("run_id", pymongo.ASCENDING)]),
            pymongo.IndexModel([("artifact_path", pymongo.ASCENDING)]),
            pymongo.IndexModel([("utc_time_created", pymongo.DESCENDING)]),
            pymongo.IndexModel([("run_id", pymongo.ASCENDING), ("artifact_path", pymongo.ASCENDING)], unique=True),
        ]
        MongoDbUtils.ensure_indexes(self.logged_models_collection, indexes)

    def _create_dataset_inputs_indexes(self):
        """Create indexes for dataset inputs collection"""
        indexes = [
            pymongo.IndexModel([("run_uuid", pymongo.ASCENDING)]),
            pymongo.IndexModel([("dataset.name", pymongo.ASCENDING)]),
            pymongo.IndexModel([("dataset.digest", pymongo.ASCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(self.dataset_inputs_collection, indexes)

    def _create_traces_indexes(self):
        """Create indexes for traces collection"""
        indexes = [
            pymongo.IndexModel([("request_id", pymongo.ASCENDING)], unique=True),
            pymongo.IndexModel([("experiment_id", pymongo.ASCENDING)]),
            pymongo.IndexModel([("timestamp_ms", pymongo.DESCENDING)]),
            pymongo.IndexModel([("execution_time_ms", pymongo.DESCENDING)]),
            pymongo.IndexModel([("status", pymongo.ASCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(self.traces_collection, indexes)

    def _create_trace_artifacts_indexes(self):
        """Create indexes for trace artifacts collection"""
        indexes = [
            pymongo.IndexModel([("request_id", pymongo.ASCENDING)], unique=True),
            pymongo.IndexModel([("experiment_id", pymongo.ASCENDING)]),
        ]
        MongoDbUtils.ensure_indexes(self.trace_artifacts_collection, indexes)
    
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
        # Get params as list of Param objects
        params = []
        for param_doc in self.params_collection.find({"run_uuid": run_id}):
            params.append(Param(param_doc["key"], param_doc["value"]))

        # Get latest metrics as list of Metric objects
        metrics = []
        for metric_doc in self.latest_metrics_collection.find({"run_uuid": run_id}):
            metrics.append(Metric(
                metric_doc["key"],
                metric_doc["value"],
                metric_doc["timestamp"],
                metric_doc["step"]
            ))

        # Get tags as list of RunTag objects
        tags = []
        for tag_doc in self.tags_collection.find({"run_uuid": run_id}):
            tags.append(RunTag(tag_doc["key"], tag_doc["value"]))

        return RunData(metrics=metrics, params=params, tags=tags)
    
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
    
    def create_experiment(self, name: str, artifact_location: str, tags: List[ExperimentTag]) -> str:
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
    def create_run(self, experiment_id: str, user_id: str, start_time: Optional[int], tags: List[RunTag], run_name: Optional[str]) -> Run:
        """Create a new run"""
        # Verify experiment exists
        experiment = self.experiments_collection.find_one({"experiment_id": experiment_id})
        if not experiment:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        # Check if experiment is active
        if experiment.get("lifecycle_stage") == LifecycleStage.DELETED:
            raise MlflowException(
                f"Experiment '{experiment_id}' is deleted",
                INVALID_STATE
            )

        run_id = self._generate_run_id()
        current_time = get_current_time_millis()

        # Generate artifact URI
        artifact_uri = None
        if self.artifact_uri:
            artifact_uri = f"{self.artifact_uri}/{experiment_id}/{run_id}/artifacts"
        elif experiment.get("artifact_location"):
            artifact_uri = f"{experiment['artifact_location']}/{run_id}/artifacts"

        run_doc = {
            "run_uuid": run_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": start_time or current_time,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "artifact_uri": artifact_uri,
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
    
    def update_run_info(self, run_id: str, run_status: Optional[RunStatus], end_time: Optional[int], run_name: Optional[str]) -> RunInfo:
        """Update run info"""
        _validate_run_id(run_id)

        update_doc = {}

        if run_status is not None:
            update_doc["status"] = RunStatus.to_string(run_status)
        if end_time is not None:
            update_doc["end_time"] = end_time
        if run_name is not None:
            update_doc["name"] = run_name

        if update_doc:
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
        
        # Update latest metric (exclude _id field to avoid immutable field error)
        update_doc = {k: v for k, v in metric_doc.items() if k != "_id"}
        self.latest_metrics_collection.update_one(
            {"run_uuid": run_id, "key": metric.key},
            {"$set": update_doc},
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
        
        self.tags_collection.update_one(
            {"run_uuid": run_id, "key": tag.key},
            {"$set": {k: v for k, v in tag_doc.items() if k != "_id"}},
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
            # Handle protobuf RepeatedScalarContainer
            exp_ids = []
            for exp_id in experiment_ids:
                if isinstance(exp_id, (list, tuple)):
                    exp_ids.extend(exp_id)
                else:
                    exp_ids.append(str(exp_id))
        else:
            exp_ids = [str(experiment_ids)] if not isinstance(experiment_ids, list) else [str(x) for x in experiment_ids]

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
    
    # Batch operations
    def log_batch(self, run_id: str, metrics: List[Metric], params: List[Param], tags: List[RunTag]) -> None:
        """Log multiple metrics, params, and tags efficiently"""
        _validate_run_id(run_id)

        # Verify run exists
        if not self.runs_collection.find_one({"run_uuid": run_id}):
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        # Batch insert metrics
        if metrics:
            metric_docs = []
            latest_metric_docs = {}

            for metric in metrics:
                _validate_metric_name(metric.key)
                metric_doc = {
                    "run_uuid": run_id,
                    "key": metric.key,
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                }
                metric_docs.append(metric_doc)

                # Keep track of latest metrics
                key = f"{run_id}_{metric.key}"
                if key not in latest_metric_docs or metric.timestamp > latest_metric_docs[key]["timestamp"]:
                    latest_metric_docs[key] = metric_doc

            if metric_docs:
                self.metrics_collection.insert_many(metric_docs)

                # Update latest metrics (exclude _id field to avoid immutable field error)
                for metric_doc in latest_metric_docs.values():
                    update_doc = {k: v for k, v in metric_doc.items() if k != "_id"}
                    self.latest_metrics_collection.update_one(
                        {"run_uuid": run_id, "key": metric_doc["key"]},
                        {"$set": update_doc},
                        upsert=True
                    )

        # Batch insert/update params
        if params:
            for param in params:
                _validate_param_name(param.key)
                param_doc = {
                    "run_uuid": run_id,
                    "key": param.key,
                    "value": param.value,
                }

                self.params_collection.update_one(
                    {"run_uuid": run_id, "key": param.key},
                    {"$set": {k: v for k, v in param_doc.items() if k != "_id"}},
                    upsert=True
                )

        # Batch insert/update tags
        if tags:
            for tag in tags:
                _validate_tag_name(tag.key)
                tag_doc = {
                    "run_uuid": run_id,
                    "key": tag.key,
                    "value": tag.value,
                }

                self.tags_collection.update_one(
                    {"run_uuid": run_id, "key": tag.key},
                    {"$set": {k: v for k, v in tag_doc.items() if k != "_id"}},
                    upsert=True
                )

    # Model logging methods
    def record_logged_model(self, run_id: str, mlflow_model: Dict[str, Any]) -> None:
        """Record a logged model"""
        _validate_run_id(run_id)

        # Verify run exists
        if not self.runs_collection.find_one({"run_uuid": run_id}):
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        model_doc = {
            "run_id": run_id,
            "artifact_path": mlflow_model.get("artifact_path"),
            "utc_time_created": mlflow_model.get("utc_time_created", get_current_time_millis()),
            "flavors": mlflow_model.get("flavors", {}),
            "model_uuid": mlflow_model.get("model_uuid"),
            "model_size_bytes": mlflow_model.get("model_size_bytes"),
        }

        # Use update_one to handle duplicate models (exclude _id field)
        self.logged_models_collection.update_one(
            {"run_id": run_id, "artifact_path": model_doc["artifact_path"]},
            {"$set": {k: v for k, v in model_doc.items() if k != "_id"}},
            upsert=True
        )

    def log_inputs(self, run_id: str, datasets: List[DatasetInput]) -> None:
        """Log input datasets"""
        _validate_run_id(run_id)

        # Verify run exists
        if not self.runs_collection.find_one({"run_uuid": run_id}):
            raise MlflowException(
                f"Run '{run_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        if datasets:
            dataset_docs = []
            for dataset in datasets:
                dataset_doc = {
                    "run_uuid": run_id,
                    "dataset": {
                        "name": dataset.dataset.name,
                        "digest": dataset.dataset.digest,
                        "source_type": dataset.dataset.source_type,
                        "source": dataset.dataset.source,
                        "schema": dataset.dataset.schema.to_dict() if dataset.dataset.schema else None,
                        "profile": dataset.dataset.profile.to_dict() if dataset.dataset.profile else None,
                    },
                    "tags": [{"key": tag.key, "value": tag.value} for tag in dataset.tags] if dataset.tags else [],
                }
                dataset_docs.append(dataset_doc)

            # Remove existing datasets for this run and insert new ones
            self.dataset_inputs_collection.delete_many({"run_uuid": run_id})
            self.dataset_inputs_collection.insert_many(dataset_docs)

    # Trace methods (basic implementation)
    def start_trace(self, experiment_id: str, timestamp_ms: int, request_metadata: Dict[str, Any],
                   tags: Dict[str, str]) -> TraceInfoV2:
        """Start a new trace"""
        # Verify experiment exists
        if not self.experiments_collection.find_one({"experiment_id": experiment_id}):
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        request_id = str(uuid.uuid4())

        # Add artifact location tag for file system storage
        trace_tags = tags or {}
        # Use file system path for artifacts (standard MLflow behavior)
        artifact_path = f"./mlflow-artifacts/traces/{request_id}"
        trace_tags["mlflow.artifactLocation"] = artifact_path

        trace_doc = {
            "request_id": request_id,
            "experiment_id": experiment_id,
            "timestamp_ms": timestamp_ms,
            "execution_time_ms": None,
            "status": TraceStatus.IN_PROGRESS.name,
            "request_metadata": request_metadata or {},
            "tags": trace_tags,
        }

        self.traces_collection.insert_one(trace_doc)

        return TraceInfoV2(
            request_id=request_id,
            experiment_id=experiment_id,
            timestamp_ms=timestamp_ms,
            execution_time_ms=None,
            status=TraceStatus.IN_PROGRESS,
            request_metadata=request_metadata or {},
            tags=trace_tags,
        )

    def end_trace(self, request_id: str, timestamp_ms: int, status: TraceStatus,
                 request_metadata: Dict[str, Any], tags: Dict[str, str],
                 spans_data: Dict[str, Any] = None) -> TraceInfoV2:
        """End a trace"""
        trace_doc = self.traces_collection.find_one({"request_id": request_id})
        if not trace_doc:
            raise MlflowException(
                f"Trace '{request_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        execution_time_ms = timestamp_ms - trace_doc["timestamp_ms"]

        update_doc = {
            "execution_time_ms": execution_time_ms,
            "status": status.name,
            "request_metadata": {**trace_doc.get("request_metadata", {}), **(request_metadata or {})},
            "tags": {**trace_doc.get("tags", {}), **(tags or {})},
        }

        self.traces_collection.update_one(
            {"request_id": request_id},
            {"$set": update_doc}
        )

        # Store trace spans data as artifact if provided
        if spans_data:
            try:
                # Save artifact to file system (standard MLflow approach)
                artifact_path = self._save_trace_artifact_to_file(request_id, spans_data, trace_doc["experiment_id"], update_doc)

                # Store the file path in MongoDB for reference
                self.traces_collection.update_one(
                    {"request_id": request_id},
                    {"$set": {"artifact_file_path": artifact_path}}
                )
                _logger.info(f"Stored trace artifact to file: {artifact_path}")
            except Exception as e:
                _logger.warning(f"Failed to store trace artifact for {request_id}: {e}")

        return TraceInfoV2(
            request_id=request_id,
            experiment_id=trace_doc["experiment_id"],
            timestamp_ms=trace_doc["timestamp_ms"],
            execution_time_ms=execution_time_ms,
            status=status,
            request_metadata=update_doc["request_metadata"],
            tags=update_doc["tags"],
        )

    def get_trace_info(self, request_id: str) -> TraceInfoV2:
        """Get trace info"""
        trace_doc = self.traces_collection.find_one({"request_id": request_id})
        if not trace_doc:
            raise MlflowException(
                f"Trace '{request_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

        return TraceInfoV2(
            request_id=trace_doc["request_id"],
            experiment_id=trace_doc["experiment_id"],
            timestamp_ms=trace_doc["timestamp_ms"],
            execution_time_ms=trace_doc.get("execution_time_ms"),
            status=TraceStatus[trace_doc["status"]],
            request_metadata=trace_doc.get("request_metadata", {}),
            tags=trace_doc.get("tags", {}),
        )

    def set_trace_tag(self, request_id: str, key: str, value: str) -> None:
        """Set a trace tag"""
        result = self.traces_collection.update_one(
            {"request_id": request_id},
            {"$set": {f"tags.{key}": value}}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Trace '{request_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

    def delete_trace_tag(self, request_id: str, key: str) -> None:
        """Delete a trace tag"""
        result = self.traces_collection.update_one(
            {"request_id": request_id},
            {"$unset": {f"tags.{key}": ""}}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Trace '{request_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )

    def search_traces(self, experiment_ids, filter_string=None,
                     max_results=100, order_by=None, page_token=None,
                     model_id=None, sql_warehouse_id=None):
        """Search traces"""
        # Add debug logging to see if this method is being called
        _logger.info(f"🔍 search_traces called with experiment_ids: {experiment_ids}, type: {type(experiment_ids)}")
        _logger.info(f"🔍 filter_string: {filter_string}, max_results: {max_results}")
        _logger.info(f"🔍 order_by: {order_by}, page_token: {page_token}")

        # Handle None or empty experiment_ids
        if not experiment_ids:
            _logger.warning("🔍 No experiment_ids provided, returning empty results")
            return [], None

        # Convert experiment_ids to list if needed
        # Handle protobuf RepeatedScalarContainer specifically
        if hasattr(experiment_ids, '__len__') and len(experiment_ids) > 0:
            # This handles protobuf containers and regular lists
            exp_ids = [str(exp_id) for exp_id in experiment_ids]
        elif isinstance(experiment_ids, str):
            exp_ids = [experiment_ids]
        elif experiment_ids:
            exp_ids = [str(experiment_ids)]
        else:
            exp_ids = []

        _logger.info(f"🔍 Converted experiment_ids to: {exp_ids}")

        if not exp_ids:
            _logger.warning("🔍 No valid experiment_ids after conversion, returning empty results")
            return [], None

        query = {"experiment_id": {"$in": exp_ids}}

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
            sort_criteria = [("timestamp_ms", pymongo.DESCENDING)]

        # Execute query
        cursor = self.traces_collection.find(query).sort(sort_criteria)

        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))

        traces = []
        for doc in cursor.limit(max_results):
            trace_info = TraceInfoV2(
                request_id=doc["request_id"],
                experiment_id=doc["experiment_id"],
                timestamp_ms=doc["timestamp_ms"],
                execution_time_ms=doc.get("execution_time_ms"),
                status=TraceStatus[doc["status"]],
                request_metadata=doc.get("request_metadata", {}),
                tags=doc.get("tags", {}),
            )
            traces.append(trace_info)

        # Check if there are more results
        next_page_token = None
        if len(traces) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)

        return traces, next_page_token

    def search_logged_models(self, experiment_ids: List[str],
                           filter_string: Optional[str] = None,
                           datasets: Optional[List[Dict[str, Any]]] = None,
                           max_results: Optional[int] = None,
                           order_by: Optional[List[Dict[str, Any]]] = None,
                           page_token: Optional[str] = None):
        """Search logged models"""
        from mlflow.entities.logged_model import LoggedModel

        query = {}

        # Set default max_results if not provided
        if max_results is None:
            max_results = 100

        if experiment_ids:
            # Get run IDs for the specified experiments
            if hasattr(experiment_ids, '__iter__') and not isinstance(experiment_ids, str):
                # Handle protobuf RepeatedScalarContainer
                exp_ids = []
                for exp_id in experiment_ids:
                    if isinstance(exp_id, (list, tuple)):
                        exp_ids.extend(exp_id)
                    else:
                        exp_ids.append(str(exp_id))
            else:
                exp_ids = [str(experiment_ids)] if not isinstance(experiment_ids, list) else [str(x) for x in experiment_ids]

            run_docs = self.runs_collection.find(
                {"experiment_id": {"$in": exp_ids}},
                {"run_uuid": 1}
            )
            run_ids = [doc["run_uuid"] for doc in run_docs]
            query["run_id"] = {"$in": run_ids}

        # Handle datasets filter (simplified implementation)
        if datasets:
            # This would need more sophisticated implementation based on dataset requirements
            pass

        # Apply ordering
        sort_criteria = []
        if order_by:
            for order_item in order_by:
                if isinstance(order_item, dict):
                    # Handle dictionary format: {"key": "field_name", "is_ascending": False}
                    field = order_item.get("key", "utc_time_created")
                    is_ascending = order_item.get("is_ascending", False)
                    direction = pymongo.ASCENDING if is_ascending else pymongo.DESCENDING
                    sort_criteria.append((field, direction))
                elif isinstance(order_item, str):
                    # Handle string format: "field_name DESC" or "field_name ASC"
                    if "DESC" in order_item:
                        field = order_item.replace(" DESC", "")
                        sort_criteria.append((field, pymongo.DESCENDING))
                    else:
                        field = order_item.replace(" ASC", "")
                        sort_criteria.append((field, pymongo.ASCENDING))
        else:
            sort_criteria = [("utc_time_created", pymongo.DESCENDING)]

        # Execute query
        cursor = self.logged_models_collection.find(query).sort(sort_criteria)

        # Apply pagination
        if page_token:
            cursor = cursor.skip(int(page_token))

        models = []
        for doc in cursor.limit(max_results):
            # Create LoggedModel entity
            try:
                logged_model = LoggedModel(
                    artifact_path=doc["artifact_path"],
                    flavors=doc.get("flavors", {}),
                    utc_time_created=doc["utc_time_created"],
                    run_id=doc["run_id"],
                    model_uuid=doc.get("model_uuid"),
                    model_size_bytes=doc.get("model_size_bytes")
                )
                models.append(logged_model)
            except Exception as e:
                # Fallback to dict if LoggedModel constructor fails
                _logger.warning(f"Failed to create LoggedModel entity: {e}")
                models.append({
                    "run_id": doc["run_id"],
                    "artifact_path": doc["artifact_path"],
                    "utc_time_created": doc["utc_time_created"],
                    "flavors": doc.get("flavors", {}),
                    "model_uuid": doc.get("model_uuid"),
                    "model_size_bytes": doc.get("model_size_bytes"),
                })

        # Check if there are more results
        next_page_token = None
        if len(models) == max_results:
            next_page_token = str((int(page_token) if page_token else 0) + max_results)

        return PagedList(models, next_page_token)

    def search_datasets(self, experiment_ids: List[str] = None,
                       filter_string: str = None, max_results: int = 100,
                       order_by: List[str] = None, page_token: str = None):
        """Search datasets (basic implementation)"""
        # This method might be called by newer MLflow versions
        # Return empty results for now
        return PagedList([], None)

    def _search_datasets(self, experiment_ids: List[str]):
        """Search datasets for MLflow server (internal method)"""
        # This is the method the MLflow server actually calls
        # For now, return empty list since we don't have dataset summaries implemented
        # In a full implementation, this would return DatasetSummary objects
        try:
            # Validate experiment_ids exist
            if experiment_ids:
                for exp_id in experiment_ids:
                    if not self.experiments_collection.find_one({"experiment_id": str(exp_id)}):
                        _logger.warning(f"Experiment {exp_id} not found")

            # Return empty list for now - could be enhanced to return actual dataset summaries
            return []
        except Exception as e:
            _logger.error(f"Error in _search_datasets: {e}")
            return []

    # Trace artifact methods (file system based)
    def _save_trace_artifact_to_file(self, request_id: str, spans_data: dict, experiment_id: str, trace_metadata: dict) -> str:
        """Save trace artifact data to file system and return file path"""
        import json
        from pathlib import Path

        try:
            # Create artifact directory structure
            artifact_dir = Path("./mlflow-artifacts/traces") / request_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Create the trace artifact data in the same format as file store
            artifact_data = {
                "experiment_id": experiment_id,
                "spans": spans_data.get("spans", []),
                "request_metadata": trace_metadata.get("request_metadata", {}),
                "tags": trace_metadata.get("tags", {})
            }

            # Save to JSON file (same format as MLflow file store)
            artifact_file = artifact_dir / "traces.json"
            with open(artifact_file, "w") as f:
                json.dump(artifact_data, f, indent=2)

            # Return absolute path for storage in MongoDB
            return str(artifact_file.absolute())

        except Exception as e:
            _logger.error(f"Error saving trace artifact to file for {request_id}: {e}")
            raise MlflowException(
                f"Failed to save trace artifact to file: {e}",
                INTERNAL_ERROR
            )

    def get_trace_artifact_path(self, request_id: str) -> str:
        """Get the file path for trace artifact"""
        try:
            trace_doc = self.traces_collection.find_one({"request_id": request_id})

            if not trace_doc:
                raise MlflowException(
                    f"Trace '{request_id}' not found",
                    RESOURCE_DOES_NOT_EXIST
                )

            # Check if artifact file path is stored
            artifact_path = trace_doc.get("artifact_file_path")
            if artifact_path:
                return artifact_path

            # Fallback: construct expected path
            return f"./mlflow-artifacts/traces/{request_id}/traces.json"

        except MlflowException:
            raise
        except Exception as e:
            _logger.error(f"Error getting trace artifact path for {request_id}: {e}")
            raise MlflowException(
                f"Failed to get trace artifact path: {e}",
                INTERNAL_ERROR
            )

    def store_trace_artifact(self, request_id: str, artifact_data: dict) -> None:
        """Store trace artifact data to file system (for backward compatibility)"""
        try:
            # Save to file system
            artifact_path = self._save_trace_artifact_to_file(
                request_id,
                {"spans": artifact_data.get("spans", [])},
                artifact_data.get("experiment_id"),
                artifact_data
            )

            # Update MongoDB with file path
            self.traces_collection.update_one(
                {"request_id": request_id},
                {"$set": {"artifact_file_path": artifact_path}}
            )

            _logger.info(f"Stored trace artifact to file: {artifact_path}")

        except Exception as e:
            _logger.error(f"Error storing trace artifact for {request_id}: {e}")
            raise MlflowException(
                f"Failed to store trace artifact: {e}",
                INTERNAL_ERROR
            )

    def get_trace_artifact(self, request_id: str) -> dict:
        """Retrieve trace artifact data from file system"""
        import json
        from pathlib import Path

        try:
            # Get the artifact file path
            artifact_path = self.get_trace_artifact_path(request_id)

            # Read from file
            if not Path(artifact_path).exists():
                raise MlflowException(
                    f"Trace artifact file not found: {artifact_path}",
                    RESOURCE_DOES_NOT_EXIST
                )

            with open(artifact_path, "r") as f:
                artifact_data = json.load(f)

            return artifact_data

        except MlflowException:
            raise
        except Exception as e:
            _logger.error(f"Error retrieving trace artifact for {request_id}: {e}")
            raise MlflowException(
                f"Failed to retrieve trace artifact: {e}",
                INTERNAL_ERROR
            )

    # Additional utility methods
    def get_metric_history_bulk(self, run_ids: List[str], metric_key: str) -> List[MetricWithRunId]:
        """Get metric history for multiple runs"""
        metrics = []
        for doc in self.metrics_collection.find(
            {"run_uuid": {"$in": run_ids}, "key": metric_key}
        ).sort("timestamp", pymongo.ASCENDING):
            metrics.append(MetricWithRunId(
                key=doc["key"],
                value=doc["value"],
                timestamp=doc["timestamp"],
                step=doc["step"],
                run_id=doc["run_uuid"]
            ))

        return metrics

    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag"""
        _validate_run_id(run_id)
        _validate_tag_name(key)

        result = self.tags_collection.delete_one({"run_uuid": run_id, "key": key})
        if result.deleted_count == 0:
            raise MlflowException(
                f"Tag '{key}' not found for run '{run_id}'",
                RESOURCE_DOES_NOT_EXIST
            )

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Set an experiment tag"""
        _validate_tag_name(tag.key)

        result = self.experiments_collection.update_one(
            {"experiment_id": experiment_id},
            {"$push": {"tags": {"key": tag.key, "value": tag.value}}}
        )
        if result.matched_count == 0:
            raise MlflowException(
                f"Experiment '{experiment_id}' not found",
                RESOURCE_DOES_NOT_EXIST
            )
