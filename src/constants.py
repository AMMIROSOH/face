import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV") if os.getenv("APP_ENV") is not None else "dev"
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))

GLOBAL_MESSAGE_LENGTH = 1
MAX_PEOPLE = 20
FRAME_SHAPE_GUI = (720, 1280, 3)