import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV") if os.getenv("APP_ENV") is not None else "dev"
SYNC_FPS = os.getenv("SYNC_FPS") == "True"
CAMERA_URL = os.getenv("CAMERA_URL")
GLOBAL_MESSAGE_LENGTH = 1
MAX_PEOPLE = 10