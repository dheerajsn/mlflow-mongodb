#!/usr/bin/env python3

"""Debug script to test prompt filtering."""

import sys

# Add local modules to path
sys.path.insert(0, '/Users/dheerajnagpal/Projects/mlflow')
from mlflow_mongodb.model_registry.mongodb_store import MongoDbModelRegistryStore

def test_prompt_filtering():
    """Test prompt filtering specifically."""
    
    print("=== Testing Prompt Filtering ===\n")
    
    from mlflow.utils.search_utils import SearchModelVersionUtils
    from mlflow.prompt.registry_utils import add_prompt_filter_string
    
    store = MongoDbModelRegistryStore("mongodb://username:passwordlocalhost:27017/mlflow")
    
    # Get all versions
    all_versions = []
    for doc in store.model_versions_collection.find():
        entity = store._model_version_doc_to_entity(doc)
        all_versions.append(entity)
        
    print(f"Retrieved {len(all_versions)} versions from MongoDB")
    for v in all_versions:
        print(f"  - {v.name} v{v.version}")
    print()
    
    # Test original filter
    filter_string = "name='p1'"
    print(f"Original filter: '{filter_string}'")
    
    # Test add_prompt_filter_string
    modified_filter = add_prompt_filter_string(filter_string, is_prompt=False)
    print(f"Modified filter: '{modified_filter}'")
    
    # Test filtering with modified filter
    try:
        filtered = SearchModelVersionUtils.filter(all_versions, modified_filter)
        print(f"Filtered results: {len(filtered)} versions")
        for v in filtered:
            print(f"  - {v.name} v{v.version}")
    except Exception as e:
        print(f"Error filtering: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test with no prompt filtering
    print("Testing without prompt filtering:")
    try:
        filtered = SearchModelVersionUtils.filter(all_versions, filter_string)
        print(f"Filtered results (no prompt filter): {len(filtered)} versions")
        for v in filtered:
            print(f"  - {v.name} v{v.version}")
    except Exception as e:
        print(f"Error filtering: {e}")

if __name__ == "__main__":
    test_prompt_filtering()
