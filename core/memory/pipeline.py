class MemoryPipeline:
 
    def __init__(self, core, event_bus):
        self.core = core
        self.event_bus = event_bus
 
    def process(self, memory):
        """
        Central memory processing pipeline.
        Future extensions:
        - enrichment
        - graph linking
        - scoring
        """
 
        self.core.store(memory)
 
        self.event_bus.emit("memory_stored", memory)
 
        return memory