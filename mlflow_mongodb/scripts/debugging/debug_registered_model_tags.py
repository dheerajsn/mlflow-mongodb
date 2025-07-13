#!/usr/bin/env python3

"""Debug script for registered model tags loading."""

import os
import sys
from pymongo import MongoClient

# Add local modules to path
sys.path.insert(0, '/Users/dheerajnagpal/Projects/mlflow')
from mlflow_mongodb.model_registry.mongodb_store import MongoDbModelRegistryStore

def debug_registered_model_tags():
    """Debug registered model tags loading."""
    
    # Initialize MongoDB store with authentication
    store = MongoDbModelRegistryStore("mongodb://username:passwordlocalhost:27017/mlflow")
    
    print("=== Debug registered model tags loading ===\n")
    
    # Test 1: Check collection names
    print("Test 1: Check collection names")
    print(f"Registered models collection: {store.registered_models_collection.name}")
    print(f"Registered model tags collection: {store.registered_model_tags_collection.name}")
    print()
    
    # Test 2: Direct collection query
    print("Test 2: Direct collection query for p1 tags")
    try:
        tags = list(store.registered_model_tags_collection.find({"name": "p1"}))
        print(f"Found {len(tags)} tags for p1:")
        for tag in tags:
            print(f"  {tag['key']}: {tag['value']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Test 3: Check what's in the tags collection
    print("Test 3: Check all tags in collection")
    try:
        all_tags = list(store.registered_model_tags_collection.find())
        print(f"Total tags in collection: {len(all_tags)}")
        for tag in all_tags:
            print(f"  {tag['name']}: {tag['key']} = {tag['value']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Test 4: Test the _registered_model_doc_to_entity method directly
    print("Test 4: Test _registered_model_doc_to_entity method")
    try:
        # Get p1 document
        p1_doc = store.registered_models_collection.find_one({"name": "p1"})
        print(f"p1 document: {p1_doc}")
        
        # Test the conversion
        if p1_doc:
            entity = store._registered_model_doc_to_entity(p1_doc)
            print(f"Converted entity:")
            print(f"  Name: {entity.name}")
            print(f"  Tags: {len(entity.tags)} tags")
            for tag in entity.tags:
                print(f"    {tag.key}: {tag.value}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 5: Check if there are any permission issues
    print("Test 5: Check database permissions")
    try:
        # Try to insert a test tag
        test_tag = {"name": "test_model", "key": "test_key", "value": "test_value"}
        store.registered_model_tags_collection.insert_one(test_tag)
        print("✓ Can insert tags")
        
        # Try to read it back
        found_tag = store.registered_model_tags_collection.find_one({"name": "test_model"})
        if found_tag:
            print("✓ Can read tags back")
        else:
            print("✗ Cannot read tags back")
            
        # Clean up
        store.registered_model_tags_collection.delete_one({"name": "test_model"})
        print("✓ Can delete tags")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

if __name__ == "__main__":
    debug_registered_model_tags()
