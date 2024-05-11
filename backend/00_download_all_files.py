import os
import boto3
from bucket import ACCESS_KEY, SECRET_KEY, REGION_NAME, BUCKET_NAME, ENDPOINT_URL
from botocore.exceptions import NoCredentialsError

s3_client = boto3.client(
    "s3",
    region_name=REGION_NAME,
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

LOCAL_DIR = "../data/"


def main() -> None:
    """
    Download all necessary files for a project from the S3 bucket to a local directory.

    Returns:
        None
    """
    os.makedirs(LOCAL_DIR, exist_ok=True)
    # For factors data
    download_folder("factors_data", LOCAL_DIR)
    # For 01_obtain_tickers.py
    download_folder("obtain_tickers", LOCAL_DIR)
    # For 02_obtain_data.py
    download_folder("obtain_data", LOCAL_DIR)
    # For 03_ranking_equities.py
    download_folder("ranking_equities", LOCAL_DIR)
    # For 04_studying_models.py
    download_folder("studying_models", LOCAL_DIR)
    # For 05_backtesting.py
    download_folder("backtest", LOCAL_DIR)


def download_folder(bucket_prefix: str, local_dir: str):
    """
    Download the contents of a specified folder directory from an AWS S3 bucket.

    Args:
        bucket_prefix (str): The prefix of the folder in the S3 bucket to be downloaded.
        local_dir (str): The local directory where the files will be downloaded.

    Returns:
        None
    """

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=bucket_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            path = os.path.join(local_dir, key)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            try:
                print(f"Downloading {key}...")
                s3_client.download_file(BUCKET_NAME, key, path)
            except NoCredentialsError:
                print("Credentials not available")
                return


if __name__ == "__main__":
    main()
