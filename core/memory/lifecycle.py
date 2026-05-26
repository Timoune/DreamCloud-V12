"""
Memory lifecycle states for future DreamCycle expansion.
"""
 
MEMORY_STATES = [
    "raw",
    "embedded",
    "indexed",
    "consolidated",
    "linked"
]
 
 
class MemoryLifecycle:
 
    @staticmethod
    def next_state(current: str):
        if current not in MEMORY_STATES:
            return "raw"
 
        idx = MEMORY_STATES.index(current)
 
        if idx + 1 >= len(MEMORY_STATES):
            return MEMORY_STATES[-1]
 
        return MEMORY_STATES[idx + 1]