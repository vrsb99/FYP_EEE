from dotenv import load_dotenv
import os

load_dotenv()

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
REGION_NAME = os.getenv("REGION_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
