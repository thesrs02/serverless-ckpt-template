from google.cloud import storage
from io import BytesIO
import os
import json
import time


def upload_images_to_gcloud(image_list):
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_dict = json.loads(credentials_json)

    storage_client = storage.Client.from_service_account_info(credentials_dict)
    bucket = storage_client.get_bucket(bucket_name)

    uploaded_urls = []

    for idx, img in enumerate(image_list):
        # Convert Pillow Image to bytes
        img_byte_arr = convert_img_to_byte_array(img)

        current_time_ms = int(time.time() * 1000)

        # Generate unique blob name
        blob_name = f"{current_time_ms}.png"
        blob = bucket.blob(blob_name)

        blob.upload_from_string(img_byte_arr, content_type="image/png")
        public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        uploaded_urls.append(public_url)

        return uploaded_urls


def convert_img_to_byte_array(img):
    img_byte_arr = bytearray()
    with BytesIO() as output:
        img.save(output, format="PNG")
        img_byte_arr = output.getvalue()
    return img_byte_arr
