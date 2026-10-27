import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
import boto3
import yaml


def _json_load_s3_cp(file_path):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch JSON from S3: {stderr.decode().strip()}")

    # Return raw bytes; parsing happens in json_load to support JSON and JSONL
    return stdout

def _jsonl_load_s3_cp(file_path):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch JSONL from S3: {stderr.decode().strip()}")

    content = stdout.decode("utf-8")
    return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def jsonl_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote jsonl.")
        return _jsonl_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries

def json_load(file_path):
    if file_path.startswith("s3"):
            logging.info("Loading remote json.")
            raw_bytes = _json_load_s3_cp(file_path)
            text = raw_bytes.decode("utf-8")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # Decide parsing mode based on extension; fallback if needed
    lower_path = str(file_path).lower()
    is_jsonl_ext = lower_path.endswith(".jsonl") or lower_path.endswith(".ndjson")
    if is_jsonl_ext:
        lines = [line for line in text.splitlines() if line.strip()]
        out = [json.loads(line) for line in lines]
    else:
        # Try whole-document JSON
        out = json.loads(text)

    return out

def _list_directory_s3_ls(dir_path):
    cmd = f"aws s3 ls {dir_path.rstrip('/') + '/'}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to list S3 directory: {stderr.decode().strip()}")

    items = []
    for line in stdout.decode("utf-8").strip().split("\n"):
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 1:
                if "PRE" in parts:
                    # Directory (prefix)
                    dirname = parts[-1].rstrip("/")
                    items.append(dirname)
                elif len(parts) >= 4:
                    # File
                    filename = " ".join(parts[3:])  # Handle filenames with spaces
                    items.append(filename)
    return items


def list_directory(dir_path):
    if dir_path.startswith("s3"):
        return _list_directory_s3_ls(dir_path)
    return os.listdir(dir_path)


def _is_dir_s3_ls(dir_path):
    """Check if an S3 path is a directory by trying to list it."""
    try:
        _list_directory_s3_ls(dir_path)
        return True
    except RuntimeError:
        return False


def is_dir(path):
    if path.startswith("s3"):
        return _is_dir_s3_ls(path)
    return os.path.isdir(path)


def _yaml_load_s3_cp(file_path):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to fetch YAML from S3: {stderr.decode().strip()}")

    return yaml.safe_load(io.BytesIO(stdout))


def yaml_load(file_path):
    if file_path.startswith("s3"):
        logging.info("Loading remote yaml.")
        return _yaml_load_s3_cp(file_path)
    with open(file_path, "r") as f:
        out = yaml.safe_load(f)
    return out


def list_directory_recursive(dir_path):
    """Get all S3 objects under a prefix efficiently using pagination."""
    assert dir_path.startswith("s3"), "Only S3 paths are supported for now"
    s3_client = boto3.client("s3")
    bucket, prefix = parse_s3_path(dir_path)
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = set()

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Store relative path from prefix
                    key = obj["Key"]
                    if key.startswith(prefix):
                        relative_key = key[len(prefix) :]
                        objects.add(relative_key.lstrip("/"))
    except Exception as e:
        print(f"Error listing directory {dir_path}: {e}")
        pass

    return objects


def _file_exists_s3_ls(file_path):
    """Check if an S3 file exists using aws s3 ls."""
    cmd = f"aws s3 ls {file_path}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        return False

    # Parse the output to ensure it's an exact match, not a prefix match
    output = stdout.decode("utf-8").strip()
    if not output:
        return False

    expected_filename = file_path.split("/")[-1]
    for line in output.split("\n"):
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 4:
                actual_filename = " ".join(parts[3:])
                if actual_filename == expected_filename:
                    return True
    return False


def file_exists(path):
    if path.startswith("s3"):
        return _file_exists_s3_ls(path)
    return os.path.exists(path)


def parse_s3_path(s3_path: str):
    assert s3_path.startswith("s3://")
    parts = s3_path.removeprefix("s3://").split("/", 1)
    bucket = parts[0]
    directory = parts[1].removesuffix("/") if len(parts) > 1 else ""
    return bucket, directory


@contextmanager
def copy_to_temp_file(file_path):
    """
    Copy a file to a temporary file and clean it up when done.
    If the file is on s3, use aws s3 cp to copy it to a temporary file.
    If the file is on the local filesystem, use shutil.copy to copy it to a temporary file.
    """
    extension = os.path.splitext(file_path)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_path = temp_file.name
    try:
        if file_path.startswith("s3"):
            cmd = f"aws s3 cp {file_path} {temp_path}"
            subprocess.run(cmd, shell=True, check=True)
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"Failed to copy S3 file: {stderr.decode().strip()}")
        else:
            shutil.copy(file_path, temp_path)
        yield temp_path
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def remote_sync(local_dir, remote_dir):
    logging.info("Starting remote sync.")
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir, remote_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False

    logging.info("Successfully synced with S3 bucket")
    return True


def s3_prefix_has_objects(s3_dir: str) -> bool:
    """
    Cheap existence check for an S3 *directory/prefix*.
    Example: s3_dir='s3://my-bucket/base/prefix/'
    """
    assert s3_dir.startswith("s3://"), "s3_dir must start with s3://"
    bucket, prefix = parse_s3_path(s3_dir)
    # Ensure trailing slash for a real 'prefix' lookup
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp and len(resp["Contents"]) > 0


def ensure_s3_prefix_local(s3_base_dir: str, rel_prefix: str, local_dir: str) -> bool:
    """
    Ensure all objects under s3://.../<rel_prefix> exist locally at local_dir.
    - Returns True if local_dir ends up populated (either already there or after download).
    - Returns False if S3 prefix is missing or download fails.
    """
    # Build full S3 prefix
    s3_prefix = f"{s3_base_dir.rstrip('/')}/{rel_prefix.strip('/')}/"
    # Fast path: if local_dir already has files, consider it present
    if os.path.isdir(local_dir):
        try:
            if any(True for _ in os.scandir(local_dir)):
                logging.info(f"Local already present: {local_dir}")
                return True
        except FileNotFoundError:
            pass

    # Check S3 availability
    if not s3_prefix_has_objects(s3_prefix):
        logging.warning(f"Missing in S3: {s3_prefix}")
        return False

    # Download (use aws cli to match your existing patterns)
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["aws", "s3", "cp", s3_prefix, local_dir, "--recursive", "--no-progress"]
    logging.info(f"Downloading S3 prefix -> local: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Download failed: {result.stderr.decode('utf-8')}")
        return False

    return True