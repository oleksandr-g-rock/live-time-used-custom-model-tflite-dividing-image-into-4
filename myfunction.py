import time
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import argparse
import io
import picamera
import cv2

def myfunction():

  #start capturing the image from the Picamera
  with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:

    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        #you can change image size from 299, 299 to any size
        img = Image.open(stream).convert('RGB').resize((299, 299),
                                                         Image.ANTIALIAS)


        ####################################BLOCK FOR IMAGE DIVIDING START#############################
        # load image for divides
        # use numpy to convert the pil_image into a numpy array
        numpy_image = image.img_to_array(img)
        # load image to cv2
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return img

        