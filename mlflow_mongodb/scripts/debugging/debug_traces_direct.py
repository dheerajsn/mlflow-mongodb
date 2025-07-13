#!/usr/bin/env python3
"""
Debug traces directly from MongoDB
"""

import sys
import os
sys.path.insert(0, '.')

from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

def debug_traces():
    print("🔍 Debugging traces directly from MongoDB...")
    
    store = MongoDbTrackingStore('mongodb://username:passwordlocalhost:27017/mlflow')
    
    # Test our search_traces method
    print("\n1. Testing search_traces method:")
    traces, token = store.search_traces(['1752352026927959'])
    print(f"   Found {len(traces)} traces")
    
    if traces:
        trace = traces[0]
        print(f"   First trace: {trace}")
        print(f"   Request ID: {trace.request_id}")
        print(f"   Experiment ID: {trace.experiment_id}")
        print(f"   Status: {trace.status}")
        
        # Test to_proto conversion
        try:
            proto = trace.to_proto()
            print(f"   ✅ to_proto() works: {type(proto)}")
        except Exception as e:
            print(f"   ❌ to_proto() failed: {e}")
    
    # Check MongoDB directly
    print("\n2. Checking MongoDB directly:")
    traces_collection = store.traces_collection
    
    # Count total traces
    total = traces_collection.count_documents({})
    print(f"   Total traces in collection: {total}")
    
    # Find traces for our experiment
    exp_traces = list(traces_collection.find({"experiment_id": "1752352026927959"}))
    print(f"   Traces for experiment 1752352026927959: {len(exp_traces)}")
    
    if exp_traces:
        trace_doc = exp_traces[0]
        print(f"   Sample trace document:")
        for key, value in trace_doc.items():
            if key == '_id':
                continue
            print(f"     {key}: {value}")
    
    # Test the server endpoint
    print("\n3. Testing server endpoint:")
    try:
        import requests
        response = requests.get("http://localhost:5001/ajax-api/2.0/mlflow/traces?experiment_ids=1752352026927959", timeout=5)
        print(f"   Server response status: {response.status_code}")
        print(f"   Server response: {response.text}")
    except Exception as e:
        print(f"   Server test failed: {e}")

if __name__ == "__main__":
    debug_traces()
