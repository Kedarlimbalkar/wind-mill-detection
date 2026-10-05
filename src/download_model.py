"""Downloads the trained model from Azure Blob Storage on container startup.

Reads AZURE_STORAGE_CONNECTION_STRING and MODEL_BLOB_PATH from environment
variables (set as secrets in deployment, never committed to Git -- BRD NFR-4),
and saves the model to the local path used by MODEL_PATH in app.py.
"""
import os

from azure.storage.blob import BlobServiceClient

CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "wind-turbine-data")
MODEL_BLOB_PATH = os.environ.get("MODEL_BLOB_PATH", "models/turbine_v10/best.pt")
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "model/best.pt")


def download_model() -> None:
    """Download the model blob to LOCAL_MODEL_PATH if not already present."""
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Model already present at {LOCAL_MODEL_PATH}, skipping download.")
        return

    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

    blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(MODEL_BLOB_PATH)

    print(f"Downloading model from {MODEL_BLOB_PATH} to {LOCAL_MODEL_PATH}...")
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print("Model download complete.")


if __name__ == "__main__":
    download_model()