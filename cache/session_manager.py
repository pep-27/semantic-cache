class SessionManager:
    def __init__(self):
        # use a dictionary to store all session histories
        # key = session_id, value = message list
        self.sessions = {}

    def add_message(self, session_id, role, content):
        """add a message to the specified session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []  # initialize an empty list
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id):
        """get the history of a session"""
        return self.sessions.get(session_id, [])
