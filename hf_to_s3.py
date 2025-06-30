#!/usr/bin/env python3
import os
import argparse
import boto3
from huggingface_hub import HfApi, hf_hub_download

def main(repo_id, s3_bucket, s3_prefix, local_dir):
    # Prepare local cache directory
    os.makedirs(local_dir, exist_ok=True)

    # Init HF API and list parquet files
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]

    if not parquet_files:
        print(f"No .parquet files found in {repo_id}")
        return

    # Prepare S3 client, possibly with custom endpoint
    endpoint = os.environ.get("S3_ENDPOINT")
    s3_kwargs = {"endpoint_url": endpoint} if endpoint else {}
    s3 = boto3.client("s3", **s3_kwargs)

    for fname in parquet_files:
        print(f"→ Downloading {fname} from HF…")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            cache_dir=local_dir,
        )

        # Construct S3 key (prefix + filename)
        key = os.path.join(s3_prefix, fname) if s3_prefix else fname
        print(f"   Uploading to s3://{s3_bucket}/{key} …")
        s3.upload_file(local_path, s3_bucket, key)

        # Clean up
        print(f"   Deleting local copy {local_path}")
        os.remove(local_path)

    print("✅ All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all .parquet files from a HF dataset repo to S3"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo ID (e.g. 'username/dataset-name')",
    )
    parser.add_argument(
        "--s3-bucket", required=True, help="Target S3 bucket name"
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="(Optional) prefix/folder in the bucket to upload into",
    )
    parser.add_argument(
        "--local-dir",
        default="./hf_tmp",
        help="Local temp dir for downloads",
    )
    args = parser.parse_args()
    main(args.repo_id, args.s3_bucket, args.s3_prefix, args.local_dir)

