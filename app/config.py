import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
    MODEL_NAME = os.environ.get(
        "MODEL_NAME",
        "EfficientNetB0 (Coffee Disease)",
    )
    TOP_PREDICTIONS = 5
    TM_MODEL_URL = os.environ.get(
        "TM_MODEL_URL",
        "/static/tm/model.json",
    )
    TM_METADATA_URL = os.environ.get(
        "TM_METADATA_URL",
        "/static/tm/metadata.json",
    )
    COFFEE_MODEL_PATH = os.environ.get(
        "COFFEE_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "coffee_disease_efficientnetb0.keras"),
    )
    COFFEE_MODEL_INPUT_SIZE = int(os.environ.get("COFFEE_MODEL_INPUT_SIZE", "224"))
    STORE_PREDICTIONS = os.environ.get("STORE_PREDICTIONS", "true").lower() == "true"
    RECENT_PREDICTIONS_LIMIT = int(os.environ.get("RECENT_PREDICTIONS_LIMIT", "5"))
    PREDICTIONS_DB = os.environ.get(
        "PREDICTIONS_DB",
        os.path.join(INSTANCE_DIR, "predictions.db"),
    )
