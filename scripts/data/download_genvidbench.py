"""Download selected GenVidBench archives (resumable; skips finished files)."""

import time

from huggingface_hub import hf_hub_download

REPO = "jian-0/GenVidBench"
FILES = [  # (path in repo, size)
    "GenVidBench/6.7m/OpenAI_Sora.zip",   # 490 MB, fake
    "GenVidBench/6.7m/keling.zip",        # 1.4 GB, fake (Kling)
    "GenVidBench/Pair1/ms.rar",           # 2.8 GB, fake (ModelScope)
    "GenVidBench/Pair2/svd.rar",          # 4.5 GB, fake (SVD)
    "GenVidBench/Pair1/pika.rar",         # 8.4 GB, fake (Pika)
    "GenVidBench/Pair1/vript.rar",        # 49 GB, real
    "GenVidBench/Pair1_labels.txt",       # labels for Pair1
]
for i, path in enumerate(FILES, 1):
    start = time.time()
    print(f"[{i}/{len(FILES)}] downloading {path}", flush=True)
    try:
        hf_hub_download(REPO, path, repo_type="dataset", local_dir="data/genvidbench")
        print(f"[{i}/{len(FILES)}] done in {time.time() - start:.0f}s", flush=True)
    except Exception as error:
        print(f"[{i}/{len(FILES)}] FAILED {type(error).__name__}: {str(error)[:200]}", flush=True)
print("ALL DONE", flush=True)
