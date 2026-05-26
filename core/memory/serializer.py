import json
 
from core.memory.schema import Memory
 
 
def serialize_memory(memory: Memory):
 
    return json.dumps(
        memory.__dict__,
        ensure_ascii=False,
        indent=2
    )
 
 
def deserialize_memory(data):
 
    return Memory(**json.loads(data))