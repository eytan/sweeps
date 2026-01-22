"""Shared utility functions for benchmark scripts."""

from typing import Any

import numpy as np


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and arrays to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy array, numpy scalar, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)
