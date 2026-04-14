from __future__ import annotations


def tree_map(fn, obj):
    if isinstance(obj, tuple):
        return tuple(tree_map(fn, item) for item in obj)
    if isinstance(obj, list):
        return [tree_map(fn, item) for item in obj]
    if isinstance(obj, dict):
        return {key: tree_map(fn, value) for key, value in obj.items()}
    return fn(obj)
