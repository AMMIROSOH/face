import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV") if os.getenv("APP_ENV") is not None else "dev"
CAMERA_URL = os.getenv("CAMERA_URL")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))

GLOBAL_MESSAGE_LENGTH = 1
MAX_PEOPLE = 10
IMAGE_SHAPE_GUI = (720, 1280, 3)