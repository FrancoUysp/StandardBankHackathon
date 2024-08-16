# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn, firestore_fn
from firebase_admin import initialize_app, firestore, storage
import requests
from flask import send_file
from io import BytesIO
import cv2 
import numpy as np

import google.cloud.firestore

initialize_app()


@https_fn.on_request()
def on_request_example(req: https_fn.Request) -> https_fn.Response:
    # Obtain image URL from request
    
    document_id = req.args.get("document_id")
    if not document_id:
        return https_fn.Response("Document ID not provided", status=400)
    
    bags = 0.5
    
    # Save image URL to Firestore
    firestore_client: google.cloud.firestore.Client = firestore.client()

    document_ref = firestore_client.collection("potholes").document(document_id)
    image_url = document_ref.get().to_dict().get("image")

    document_ref.update({
        "bags": bags
    })

    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        return https_fn.Response("Failed to fetch image", status=400)
    
    # Convert the image data to a format OpenCV can read
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Optionally, you can save the image to a file or process it further
    # For example, save the image to a file
    #cv2.imwrite('/tmp/fetched_image.jpg', image)
    
    # Return the image as a response
    _, buffer = cv2.imencode('.jpg', image)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')
    
    # return send_file(BytesIO(response.content), mimetype='image/jpeg')
    
    # return https_fn.Response(f"Image URL: {image_url}, Document ID: {document_id}")

@firestore_fn.on_document_created(document="potholes/{pushId}")
def estimate_bags(event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None]) -> None:
    """Listens for new documents to be added to /messages. If the document has
    an "original" field, creates an "uppercase" field containg the contents of
    "original" in upper case."""

    # Get the value of "original" if it exists.
    if event.data is None:
        return
    try:
        image_url = event.data.get("image")
    except KeyError:
        # No "original" field, so do nothing.
        return
    
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        return https_fn.Response("Failed to fetch image", status=400)
    
    # Convert the image data to a format OpenCV can read
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    print(image)
    
    # Optionally, you can save the image to a file or process it further
    # For example, save the image to a file
    #cv2.imwrite('/tmp/fetched_image.jpg', image)

    # Set the "uppercase" field.
    event.data.reference.update({"bags": 0.5})