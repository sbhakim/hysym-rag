import json
import os
from datetime import datetime

class QueryLogger:
    def __init__(self, log_file="logs/query_log.json"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

    def log_query(self, query, result, source):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result,
            "source": source,
        }
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(log_entry)

        with open(self.log_file, "w") as file:
            json.dump(logs, file, indent=4)
