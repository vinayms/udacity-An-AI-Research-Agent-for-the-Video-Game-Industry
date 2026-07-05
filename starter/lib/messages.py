# starter/lib/messages.py

class MessageHistory:
    def __init__(self, system_prompt=None):
        self.messages = []
        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self):
        return self.messages

    def clear(self, system_prompt=None):
        self.messages = []
        if system_prompt:
            self.add_system_message(system_prompt)
