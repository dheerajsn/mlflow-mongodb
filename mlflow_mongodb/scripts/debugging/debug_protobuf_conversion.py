#!/usr/bin/env python3
"""
Debug the protobuf conversion issue
"""

import sys
import os

def debug_conversion_step_by_step():
    """Debug the conversion step by step"""
    print("🔍 Debugging conversion step by step...")
    
    try:
        # Create protobuf message
        from mlflow.protos.service_pb2 import SearchTraces
        
        request_message = SearchTraces()
        request_message.experiment_ids.extend(['1752352026927959'])
        
        experiment_ids = request_message.experiment_ids
        print(f"📝 Original experiment_ids: {experiment_ids}")
        print(f"📝 Type: {type(experiment_ids)}")
        print(f"📝 Has __iter__: {hasattr(experiment_ids, '__iter__')}")
        print(f"📝 Is string: {isinstance(experiment_ids, str)}")
        print(f"📝 Length: {len(experiment_ids)}")
        
        # Test our conversion logic
        if not experiment_ids:
            print("❌ experiment_ids is falsy!")
            return False
        
        # Convert experiment_ids to list if needed
        if hasattr(experiment_ids, '__iter__') and not isinstance(experiment_ids, str):
            print("✅ Taking iterable path")
            # Handle protobuf RepeatedScalarContainer and other iterables
            exp_ids = []
            for i, exp_id in enumerate(experiment_ids):
                print(f"   Item {i}: {exp_id}, type: {type(exp_id)}")
                if isinstance(exp_id, (list, tuple)):
                    print(f"   -> Extending with: {[str(x) for x in exp_id]}")
                    exp_ids.extend(str(x) for x in exp_id)
                else:
                    print(f"   -> Appending: {str(exp_id)}")
                    exp_ids.append(str(exp_id))
        else:
            print("✅ Taking single item path")
            exp_ids = [str(experiment_ids)]
        
        print(f"✅ Final exp_ids: {exp_ids}")
        print(f"✅ exp_ids length: {len(exp_ids)}")
        print(f"✅ exp_ids is truthy: {bool(exp_ids)}")
        
        if not exp_ids:
            print("❌ exp_ids is empty after conversion!")
            return False
        
        # Test MongoDB query
        query = {"experiment_id": {"$in": exp_ids}}
        print(f"✅ MongoDB query: {query}")
        
        # Test with actual MongoDB
        sys.path.insert(0, '.')
        from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
        
        mongodb_store = MongoDbTrackingStore("mongodb://username:passwordlocalhost:27017/mlflow")
        traces_collection = mongodb_store.traces_collection
        
        count = traces_collection.count_documents(query)
        print(f"✅ MongoDB query result: {count} traces")
        
        return count > 0
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_search_traces():
    """Test a minimal version of search_traces"""
    print("\n🔍 Testing minimal search_traces...")
    
    try:
        # Create protobuf message
        from mlflow.protos.service_pb2 import SearchTraces
        
        request_message = SearchTraces()
        request_message.experiment_ids.extend(['1752352026927959'])
        
        # Minimal conversion
        exp_ids = [str(exp_id) for exp_id in request_message.experiment_ids]
        print(f"📝 Minimal conversion result: {exp_ids}")
        
        # Test with MongoDB
        sys.path.insert(0, '.')
        from mlflow_mongodb.tracking.mongodb_store import MongoDbTrackingStore
        
        mongodb_store = MongoDbTrackingStore("mongodb://username:passwordlocalhost:27017/mlflow")
        
        # Direct MongoDB query
        query = {"experiment_id": {"$in": exp_ids}}
        cursor = mongodb_store.traces_collection.find(query)
        
        traces = []
        for doc in cursor:
            print(f"📄 Found trace doc: {doc['request_id']}")
            
            from mlflow.entities.trace_info_v2 import TraceInfoV2
            from mlflow.entities.trace_status import TraceStatus
            
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
        
        print(f"✅ Minimal search_traces found {len(traces)} traces")
        return len(traces) > 0
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Protobuf Conversion Debug")
    print("=" * 50)
    
    # Debug step by step
    step_success = debug_conversion_step_by_step()
    
    # Test minimal version
    minimal_success = test_minimal_search_traces()
    
    if step_success and minimal_success:
        print("\n✅ Conversion works! The issue must be elsewhere.")
    else:
        print("\n❌ Conversion has issues!")


if __name__ == "__main__":
    main()
