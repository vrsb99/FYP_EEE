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


def upload_file(file_path: str, bucket: str, object_name: str) -> None:
    """
    Uploads a file to an AWS S3 bucket. It prints an error message if the upload fails, otherwise it confirms the upload success.

    Args:
        file_path (str): The path to the file on the local filesystem that needs to be uploaded.
        bucket (str): The name of the S3 bucket where the file will be uploaded.
        object_name (str): The S3 object name under which the file will be stored.

    Returns:
        None
    """

    try:
        response = s3_client.upload_file(file_path, bucket, object_name)
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
    else:
        print(f"Uploaded {file_path} to {object_name}")


def main() -> None:
    """
    Uploads specified directories and their files from a local directory to an S3 bucket. The paths are predefined,
    and each file within these paths is uploaded under a corresponding path in the S3 bucket.

    Returns:
        None
    """
    # All folder paths
    paths = [
        "backtest/pickle",
        "backtest/validation",
        "factors_data",
        "obtain_data",
        "obtain_tickers",
        "ranking_equities/expected_three",
        "ranking_equities/expected_five",
        "ranking_equities/ordinary",
        "studying_models/output_returns",
        "studying_models/port_weights",
        "studying_models/input_returns",
        "old_data",
    ]
    for path in paths:
        local_path = os.path.join(LOCAL_DIR, path)
        for filename in os.listdir(local_path):
            file_path = os.path.join(local_path, filename)
            if os.path.isfile(file_path):
                bucket_path = os.path.join(path, filename)
                upload_file(file_path, BUCKET_NAME, bucket_path)


if __name__ == "__main__":
    main()
