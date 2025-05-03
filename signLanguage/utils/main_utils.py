import base64
import cv2
import numpy as np

def decodeImage(imgString, fileName):
    """
    Decodes a base64 image string and saves it as a file.
    """
    imgData = base64.b64decode(imgString)
    with open(fileName, 'wb') as f:
        f.write(imgData)

def encodeImageIntoBase64(imagePath):
    """
    Encodes an image file into a base64 string.
    """
    with open(imagePath, "rb") as f:
        return base64.b64encode(f.read())
