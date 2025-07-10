# MongoDB MLflow Stores

A complete MongoDB backend implementation for MLflow 3.0+ tracking and model registry stores.

## Features

- **Complete MLflow Integration**: Implements both tracking and model registry stores
- **MongoDB Native**: Uses MongoDB's native features for optimal performance
- **Scalable**: Designed to handle large-scale ML workflows
- **Compatible**: Works with MLflow 3.0+ and follows the official store interface
- **Indexed**: Proper indexing for fast queries and operations
- **Prompt Support**: Ready for MLflow 3.0+ prompt management features

## Architecture

### Tracking Store (`MongoDbTrackingStore`)

The tracking store manages:
- **Experiments**: ML experiments with metadata and lifecycle management
- **Runs**: Individual experiment runs with status tracking
- **Metrics**: Time-series metrics with history support
- **Parameters**: Run parameters with validation
- **Tags**: Key-value metadata for runs and experiments
- **Artifacts**: Artifact URI management
- **Traces**: Distributed tracing support (MLflow 3.0+)

### Model Registry Store (`MongoDbModelRegistryStore`)

The model registry store manages:
- **Registered Models**: Model definitions with versioning
- **Model Versions**: Individual model versions with stages
- **Tags**: Metadata for models and versions
- **Aliases**: Named references to model versions
- **Prompts**: Prompt management (MLflow 3.0+)
- **Lifecycle**: Model promotion and archival

## MongoDB Schema

### Collections

#### Tracking Store Collections:
- `{prefix}_experiments`: Experiment metadata
- `{prefix}_runs`: Run information and status
- `{prefix}_metrics`: All metric points with history
- `{prefix}_latest_metrics`: Latest value for each metric
- `{prefix}_params`: Run parameters
- `{prefix}_tags`: Run and experiment tags

#### Model Registry Collections:
- `{prefix}_registered_models`: Model definitions
- `{prefix}_model_versions`: Model versions
- `{prefix}_model_version_tags`: Tags for model versions
- `{prefix}_registered_model_tags`: Tags for registered models
- `{prefix}_registered_model_aliases`: Model aliases

### Indexes

The implementation creates optimal indexes for:
- Unique constraints (experiment names, run IDs)
- Query performance (time-based queries, filtering)
- Sorting and pagination
- Cross-collection joins

## Installation

1. Install dependencies:
```bash
pip install pymongo mlflow pandas
```

2. Set up MongoDB with authentication:
```bash
# Start MongoDB with authentication
mongod --auth --dbpath ./data

# Or using Docker with authentication
docker run -d -p 27017:27017 --name mongodb \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:latest
```

3. Configure MLflow with authentication:
```bash
export MLFLOW_TRACKING_URI=mongodb://admin:password@localhost:27017/mlflow
export MLFLOW_REGISTRY_URI=mongodb://admin:password@localhost:27017/mlflow
```

## Configuration

### MongoDB URI Format

The stores accept MongoDB URIs in the following format:
```
mongodb://[username:password@]host[:port]/database[?options][#collection_prefix]
```

Examples:
- `mongodb://localhost:27017/mlflow` (no authentication)
- `mongodb://admin:password@localhost:27017/mlflow` (with authentication)
- `mongodb://admin:password@localhost:27017/mlflow?authSource=admin` (with auth source)
- `mongodb://admin:password@localhost:27017/mlflow#custom_prefix` (with custom collection prefix)

### Authentication Support

For MongoDB instances with authentication enabled (like your setup):
```python
# MongoDB URI with authentication
mongodb_uri = "mongodb://admin:password@localhost:27017/mlflow"

# Initialize stores
tracking_store = MongoDbTrackingStore(mongodb_uri)
model_registry_store = MongoDbModelRegistryStore(mongodb_uri)
```

The stores automatically configure authentication with:
- **authSource**: "admin" (default for admin users)
- **Proper connection pooling**: Optimized for authenticated connections
- **Secure connections**: Support for SSL/TLS connections via query parameters

## Usage

### Basic Setup

```python
from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
from mlflow_mongodb.model_registry.mongodb_store import MongoDbModelRegistryStore

# Create stores with authentication
tracking_store = MongoDbTrackingStore("mongodb://admin:password@localhost:27017/mlflow")
registry_store = MongoDbModelRegistryStore("mongodb://admin:password@localhost:27017/mlflow")
```

### With MLflow Client

```python
import mlflow
from mlflow_mongodb.registration import register_mongodb_stores

# Register MongoDB stores
register_mongodb_stores()

# Use MLflow normally
mlflow.set_tracking_uri("mongodb://localhost:27017/mlflow")
mlflow.set_registry_uri("mongodb://localhost:27017/mlflow")

# Now MLflow will use MongoDB backend
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

### Direct Store Usage

```python
from mlflow.entities import Metric, Param, RunTag

# Create experiment
experiment_id = tracking_store.create_experiment(
    name="mongodb_experiment",
    artifact_location="./artifacts",
    tags=[]
)

# Create run
run = tracking_store.create_run(
    experiment_id=experiment_id,
    user_id="user123",
    start_time=None,
    tags=[],
    run_name="mongodb_run"
)

# Log data
tracking_store.log_param(run.info.run_id, Param("lr", "0.01"))
tracking_store.log_metric(run.info.run_id, Metric("acc", 0.95, 1234567890, 0))
tracking_store.set_tag(run.info.run_id, RunTag("model", "rf"))
```

## Configuration

### Connection URI Format

```
mongodb://[username:password@]host[:port]/database[?options]
```

Examples:
- `mongodb://localhost:27017/mlflow`
- `mongodb://user:pass@localhost:27017/mlflow?authSource=admin`
- `mongodb://mongo1.example.com:27017,mongo2.example.com:27017/mlflow?replicaSet=rs0`

### Environment Variables

```bash
# Basic configuration
export MLFLOW_TRACKING_URI=mongodb://localhost:27017/mlflow
export MLFLOW_REGISTRY_URI=mongodb://localhost:27017/mlflow

# Advanced configuration
export MONGODB_COLLECTION_PREFIX=mlflow
export MONGODB_MAX_POOL_SIZE=50
export MONGODB_SERVER_SELECTION_TIMEOUT=5000
```

## Performance Considerations

### Indexing Strategy

The implementation creates comprehensive indexes:
- Unique indexes for primary keys
- Compound indexes for common query patterns
- Time-based indexes for metric history
- Text indexes for search operations

### Connection Pooling

MongoDB client is configured with:
- Connection pooling (5-50 connections)
- Automatic reconnection
- Timeout management
- Write concern for durability

### Query Optimization

- Projection to reduce data transfer
- Aggregation pipelines for complex queries
- Proper use of MongoDB operators
- Pagination for large result sets

## Error Handling

The implementation provides:
- Comprehensive error mapping to MLflow exceptions
- Proper HTTP status codes
- Detailed error messages
- Graceful degradation

## Monitoring

### Metrics to Monitor

- Connection pool usage
- Query execution times
- Index hit rates
- Document sizes
- Collection growth

### MongoDB Monitoring

```javascript
// Check collection stats
db.mlflow_experiments.stats()
db.mlflow_runs.stats()

// Monitor index usage
db.mlflow_runs.aggregate([{$indexStats: {}}])

// Check query performance
db.setProfilingLevel(2)
db.system.profile.find().limit(5).sort({wall: -1}).pretty()
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check MongoDB service status
   - Verify connection parameters
   - Check network connectivity

2. **Permission Errors**
   - Ensure database user has read/write permissions
   - Check authentication source

3. **Performance Issues**
   - Monitor index usage
   - Check query patterns
   - Consider sharding for large datasets

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable MongoDB debug logging
logging.getLogger('pymongo').setLevel(logging.DEBUG)
```

## Comparison with Other Stores

| Feature | MongoDB | PostgreSQL | SQLite | File |
|---------|---------|------------|--------|------|
| Scalability | High | Medium | Low | Low |
| Concurrency | High | Medium | Low | Low |
| Query Performance | High | High | Medium | Low |
| Setup Complexity | Medium | High | Low | Low |
| Operational Overhead | Medium | High | Low | Low |
| Cloud Native | Yes | Yes | No | No |

## Future Enhancements

- [ ] Sharding support for horizontal scaling
- [ ] Full-text search capabilities
- [ ] Real-time change streams
- [ ] Advanced aggregation pipelines
- [ ] Time-series optimizations
- [ ] Prometheus metrics export
- [ ] Kubernetes operator integration

## Contributing

1. Follow MLflow's contribution guidelines
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Test with multiple MongoDB versions

## License

This implementation follows the same license as MLflow.
