class SessionState:
 
    def __init__(self):
        self.messages = []
 
    def add_user_message(self, text):
        self.messages.append({
            "role": "user",
            "content": text
        })
 
        self._trim()
 
    def add_assistant_message(self, text):
        self.messages.append({
            "role": "assistant",
            "content": text
        })
 
        self._trim()
 
    def _trim(self):
 
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]