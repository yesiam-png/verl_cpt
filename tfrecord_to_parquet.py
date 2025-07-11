import os
import subprocess
from glob import glob

import tensorflow as tf
import pandas as pd

# --- CONFIG ---
BUCKET = "gs://permanent-us-central1-0rxn/tensorflow_datasets/code_textbook_v1_opc_tfds_wtc/python/1.0.0"
LOCAL_TMP = "../tmp_tfrecords"
OUT_DIR = "../parquets"
SHARD_COUNT = 256
GROUP_SIZE = 30

os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# 1) parsing spec
feature_description = {
    "text": tf.io.FixedLenFeature([], tf.string),
}
def _parse_fn(serialized):
    return tf.io.parse_single_example(serialized, feature_description)

def tfrecord_to_rows(path):
    """Parse one TFRecord shard into a list of {'text': str} dicts."""
    ds = tf.data.TFRecordDataset([path]).map(_parse_fn)
    rows = []
    for ex in ds:
        text = ex["text"].numpy().decode("utf-8")
        rows.append({"text": text})
    return rows


# 2) main loop over shard‐groups
chunk_idx = 0
for start in range(0, SHARD_COUNT, GROUP_SIZE):
    # figure out which shard‐ids to grab
    shard_ids = list(range(start, min(start + GROUP_SIZE, SHARD_COUNT)))
    local_paths = []
    
    # 2a) download each shard
    for sid in shard_ids:
        fname = f"python-train.tfrecord-{sid:05d}-of-00256"
        remote = f"{BUCKET}/{fname}"
        local = os.path.join(LOCAL_TMP, fname)
        print(f"↓ gsutil cp {remote} → {local}")
        subprocess.run(["gsutil", "cp", remote, local], check=True)
        local_paths.append(local)
    
    # 2b) parse & accumulate rows
    all_rows = []
    for lp in local_paths:
        print(f"   parsing {lp} …")
        all_rows.extend(tfrecord_to_rows(lp))
    
    # 2c) write one Parquet
    out_fname = os.path.join(OUT_DIR, f"python-train-part{chunk_idx:03d}.parquet")
    df = pd.DataFrame(all_rows)
    print(f"↳ writing {len(df)} rows → {out_fname}")
    df.to_parquet(out_fname, engine="pyarrow", index=False)
    
    # 2d) cleanup
    for lp in local_paths:
        os.remove(lp)
    chunk_idx += 1

print("✅ All done!")
