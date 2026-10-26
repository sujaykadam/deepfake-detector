"""Extract a random sample of clips from the downloaded GenVidBench archives.

Usage: python scripts/data/extract_genvidbench_subset.py data/genvidbench/GenVidBench data/genvidbench/subset
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
random.seed(0)
# (archive, sample size or None for all, subfolder written under out)
SOURCES = [("Pair1/vript.rar", 1500), ("Pair1/ms.rar", 400), ("Pair1/pika.rar", 400),
           ("Pair2/svd.rar", 400), ("6.7m/keling.zip", None), ("6.7m/OpenAI_Sora.zip", None)]

for archive, count in SOURCES:
    listing = subprocess.run(["bsdtar", "-tf", str(root / archive)], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    names = [n for n in listing if n.lower().endswith((".mp4", ".mov"))
             and not n.startswith("__MACOSX") and "/._" not in n]
    if count and len(names) > count:
        names = random.sample(names, count)
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as handle:
        handle.write("\n".join(names))
        handle.flush()
        subprocess.run(["bsdtar", "-xf", str(root / archive), "-C", str(out),
                        "-T", handle.name, "--fast-read"], check=True)
    print(f"{archive}: extracted {len(names)} of {len(listing)} entries", flush=True)
