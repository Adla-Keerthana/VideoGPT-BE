import os
from tempfile import NamedTemporaryFile


def save_video_file(file):
    # Save the uploaded file to a temporary location
    temp_file = NamedTemporaryFile(delete=False)
    with open(temp_file.name, "wb") as buffer:
        buffer.write(file.file.read())
    return temp_file.name
