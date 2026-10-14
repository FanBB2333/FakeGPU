"""Small CPU-only sweep example; measured wall time is not a GPU forecast."""

import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
checksum = sum(i * i for i in range(args.size))
seconds = time.perf_counter() - started
args.output.write_text(json.dumps({"seconds": seconds, "checksum": checksum}))
