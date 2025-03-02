import os

def get_unique_filename(filename: str):
    base, ext = filename.split(".")
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}.{ext}"
        counter += 1
    return new_filename