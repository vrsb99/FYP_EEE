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
    Main function that orchestrates the download of various data directories from an S3 bucket to a local directory.
    It ensures that the required local directories exist and initiates the download process for each specified folder in the S3 bucket.

    Returns:
        None: This function does not return any values.
    """

    os.makedirs(LOCAL_DIR, exist_ok=True)
    download_folder("obtain_tickers", LOCAL_DIR)
    download_folder("studying_models", LOCAL_DIR)
    download_folder("backtest/validation", LOCAL_DIR)
    download_folder("obtain_data", LOCAL_DIR)
    download_folder("old_data", LOCAL_DIR)
    download_folder("ranking_equities", LOCAL_DIR)


def download_folder(bucket_prefix: str, local_dir: str) -> None:
    """
    Downloads the contents of a specified folder from an S3 bucket to a local directory.
    It handles the creation of necessary local subdirectories and manages file downloads within those directories.

    Args:
        bucket_prefix (str): The prefix in the S3 bucket that specifies the folder to download.
        local_dir (str): The local directory path where the contents of the folder will be downloaded.

    Returns:
        None: This function does not return any values but prints download progress and error messages.
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
