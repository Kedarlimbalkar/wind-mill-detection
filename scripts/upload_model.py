"""Uploads the v10 model to Azure Blob Storage (production path).

Reads AZURE_STORAGE_CONNECTION_STRING from the environment -- never
hardcode this (BRD NFR-4).
"""
import os

from azure.storage.blob import BlobServiceClient

CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
LOCAL_PT_PATH = r"E:\Atman-project\V10\best.pt"

svc = BlobServiceClient.from_connection_string(CONNECTION_STRING)
blob = svc.get_container_client("wind-turbine-data").get_blob_client("models/turbine_v10/best.pt")

file_size = os.path.getsize(LOCAL_PT_PATH)
print(f"Uploading {LOCAL_PT_PATH} ({file_size / 1024 / 1024:.1f} MB)...")

with open(LOCAL_PT_PATH, "rb") as f:
    blob.upload_blob(
        f,
        overwrite=True,
        timeout=600,
        connection_timeout=600,
        max_concurrency=2,
    )

print("Uploaded successfully to models/turbine_v10/best.pt")