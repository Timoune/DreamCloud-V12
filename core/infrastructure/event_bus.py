class EventBus:

    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event, fn):
        self.subscribers.setdefault(event, []).append(fn)

    def emit(self, event, data):
        for fn in self.subscribers.get(event, []):
            try:
                fn(data)
            except Exception as e:
                print(f"[EventBus ERROR] {event}: {e}")