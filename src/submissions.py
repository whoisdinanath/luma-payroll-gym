import hashlib
import json


def submission_hash(answers, task_ids):
    if not isinstance(answers, list) or len(answers) != len(task_ids):
        raise ValueError("Submit exactly one answer per task")
    for item in answers:
        if not isinstance(item, dict) or set(item) != {"id", "answer"}:
            raise ValueError("Invalid answer fields")
        if not isinstance(item["id"], str) or not isinstance(item["answer"], (dict, str)):
            raise ValueError("Invalid answer types")
    if {item["id"] for item in answers} != set(task_ids):
        raise ValueError("Submit exactly one answer per task")
    encoded = json.dumps(sorted(answers, key=lambda item: item["id"]), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
