import os
import hashlib
import math
import yaml
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

def get_unique_filename(filename: str):
    """
    Get a unique filename by adding a counter to the filename.

    Args:
        filename: Original filename.

    Returns:
        Unique filename to avoid overwriting.
    """
    # split specified filename by name and extension
    base, ext = filename.split(".")
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        # count existing files with same base name
        new_filename = f"{base}_{counter}.{ext}"
        counter += 1
    return new_filename

def short_hash(obj: object):
    """Return a 5-digit hash code of specified object using md5 algorithm."""
    hash = int(hashlib.md5(str(obj).encode()).hexdigest(), 16)
    # keep only last 5 digits
    hash = hash // 10 ** (math.ceil(math.log(hash, 10)) - 5)
    return hash
