# src/utils/output_capture.py

import sys
import os
from contextlib import contextmanager
from datetime import datetime


class TeeOutput:
    def __init__(self, file_path):
        self.file = open(file_path, 'a', buffering=1)  # Line buffering
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()  # Ensure immediate write to file

    def flush(self):
        self.file.flush()
        self.stdout.flush()


@contextmanager
def capture_output(output_dir="logs"):
    """Context manager to capture all stdout/stderr to a file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"symrag_run_{timestamp}.txt")

    tee = TeeOutput(output_path)
    sys.stdout = tee
    sys.stderr = tee

    print(f"Output capture started. Saving to: {output_path}")

    try:
        yield output_path
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.file.close()
        print(f"Output capture complete: {output_path}")