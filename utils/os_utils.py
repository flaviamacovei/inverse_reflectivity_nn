import os
import hashlib
import math

def get_unique_filename(filename: str):
    base, ext = filename.split(".")
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}.{ext}"
        counter += 1
    return new_filename

def short_hash(obj: object):
    hash = int(hashlib.md5(str(obj).encode()).hexdigest(), 16)
    hash = hash // 10 ** (math.ceil(math.log(hash, 10)) - 5)
    return hash