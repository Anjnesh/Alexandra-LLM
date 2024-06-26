from model import describe_key_elements
import cv2
import numpy as np
from PIL import Image
import io

def process_a1(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    description = describe_key_elements(image)
    return description

def process_a2(image_bytes, heatmap_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    heatmap = np.array(Image.open(io.BytesIO(heatmap_bytes)).convert("RGB"))

    red_channel = heatmap[:,:,0]
    _, thresholded = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    salient_elements = []

    for contour in contours[:5]:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        roi_image = Image.fromarray(roi)
        description = describe_key_elements(roi_image)
        salient_elements.append(description)

    return salient_elements
