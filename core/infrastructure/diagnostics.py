import os
 
from config.memory_config import MEMORY_PATH
 
 
def memory_count():
 
    if not os.path.exists(MEMORY_PATH):
        return 0
 
    return len([
        f for f in os.listdir(MEMORY_PATH)
        if f.endswith(".json")
    ])