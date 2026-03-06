"""
Validates ChatML examples before they enter training.

A valid ChatML example must:
- Have a 'messages' key with a non-empty list
- Start with a 'system' role message
- Alternate user/assistant roles after the system message
- Have no empty content fields
- Use only valid roles: system, user, assistant
"""
from __future__ import annotations

import json
from pathlib import Path


def validate_example(example: dict) -> tuple[bool, str | None]:
    """
    Validates a single ChatML example dict.

    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
    if not isinstance(example, dict):
        return False, "Example must be a dict"

    if "messages" not in example:
        return False, "Missing 'messages' key"

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "messages must be a list with at least 2 elements"

    if messages[0].get("role") != "system":
        return False, "First message must have role='system'"

    valid_roles = {"system", "user", "assistant"}
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")

        if role not in valid_roles:
            return False, f"Invalid role '{role}' at index {i}"

        if not isinstance(content, str) or not content.strip():
            return False, f"Empty or non-string content at index {i}"

    # Check alternation: after system, must be user/assistant/user/assistant...
    roles_after_system = [m["role"] for m in messages[1:]]
    for i in range(len(roles_after_system) - 1):
        if roles_after_system[i] == roles_after_system[i + 1]:
            return False, (
                f"Role collision: '{roles_after_system[i]}' followed by "
                f"'{roles_after_system[i + 1]}' at position {i + 1}"
            )

    # Must start with user after system
    if roles_after_system[0] != "user":
        return False, "Second message must have role='user'"

    # Must end with assistant
    if roles_after_system[-1] != "assistant":
        return False, "Last message must have role='assistant'"

    return True, None


def validate_dataset_file(path: str | Path) -> dict:
    """
    Validates an entire JSONL file.

    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "errors": [{"line": int, "error": str}]
        }
    """
    results: dict = {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            results["total"] += 1
            try:
                example = json.loads(line)
                is_valid, error = validate_example(example)
                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["errors"].append({"line": line_num, "error": error})
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append({"line": line_num, "error": f"JSON parse error: {e}"})

    return results
