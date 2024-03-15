import cv2
import numpy as np
import pytesseract
import re
from pyzbar.pyzbar import decode
import os
import threading
from PIL import Image
from flask import jsonify
from flask import Flask, request
import requests
import time
import json
from datetime import datetime
import pytz
from pytz import timezone
from twilio.rest import Client
from twilio.http.http_client import TwilioHttpClient
from twilio import twiml



app = Flask(__name__)
account_sid="AC35efeca197129b9ce0cf37bb98e4f65f"
auth_token="54bc2ed2e4a62a9412fd3f1eb2c7b539"

client = Client(account_sid, auth_token)

@app.route('/sms', methods=['POST'])
def sms():
    print('test')
    number = request.form['From']
    message_body = request.form['Body']
    resp = twiml.Response()
    resp.message("hi")
    return str(resp)


@app.route("/")
def home():
    return "Hello World"

try:
  @app.route("/image-processing", methods=['POST'])
  def imageProcessing():
      try:
        if 'file' not in request.files:
            return jsonify(error="No file part")

        file = request.files['file']

        if file.filename == '':
            return jsonify(error="No selected file")

        # Save the uploaded file
        file_path = 'uploads/' + file.filename
        # Read the image using Pillow
        image = Image.open(file)

        # Check orientation metadata
        orientation = image.getexif().get(0x0112, 1)

        # Adjust the orientation if needed
        if orientation == 3:  # Rotate 180 degrees
            image = image.rotate(180, expand=True)
        elif orientation == 6:  # Rotate 90 degrees clockwise
            image = image.rotate(-90, expand=True)
        elif orientation == 8:  # Rotate 90 degrees counterclockwise
            image = image.rotate(90, expand=True)

        # Save the adjusted image
        image.save(file_path)

        # Read the uploaded image using OpenCV
        img = cv2.imread(file_path)

        # Check if the image is successfully loaded
        if img is None:
            return jsonify(error="Failed to load the image")

        img_meter_id = cv2.resize(img, (914, 1128))
        img_meter_reading = cv2.resize(img, (512, 720))

        # Perform image processing meter id
        simplethresh_meter_id = preprocessing1(img_meter_id)
        result_white_object = detectwhitearea(img_meter_id, simplethresh_meter_id)

        # Check if a white object is found before displaying
        if result_white_object is not None:
            print("result_white_object")
        else:
            print("No white objects found.")
            return jsonify(error="Cannot Detect Image")

        # Separate upper and lower halves
        _, lower_half = seperate(result_white_object)

        meter_id = ''

        preprocessed_qrcode = preprocessing1(lower_half)
        data_qrcode = detect_qr_codes(lower_half, preprocessed_qrcode)
        if data_qrcode is None:
          print("qrcode is null")
        else:
          meter_id = data_qrcode
          print(f"qrcode: {data_qrcode}")

        preprocessed_img = preprocessing4(lower_half)
        data_barcode = detectbarcodes(lower_half, preprocessed_img)
        if data_barcode is None:
          print("barcode is null")
        else:
          meter_id = data_barcode
          print(f"barcode: {data_barcode}")

        # Perform image processing meter reading
        simplethresh_meter_reading = preprocessing1(img_meter_reading)
        result_white_object = detectwhitearea(img_meter_reading, simplethresh_meter_reading)

        # Check if a white object is found before displaying
        if result_white_object is not None:
            print("result_white_object")
        else:
            print("No white objects found.")
            return jsonify(error="Cannot Detect Image")

        # Separate upper and lower halves
        upper_half, _ = seperate(result_white_object)

        # Getting black area
        edged = preprocessing2(upper_half)

        # Detect black area in the upper half
        result_black_object = detectblackarea(upper_half, edged)
        print("result_black_object")

        # OCR
        # Preprocess for OCR
        adaptivethreshold = preprocessing3(result_black_object)

        # Extract digits using OCR
        all_extracted_digits = OCR(result_black_object, adaptivethreshold)
        print(f"Extracted Digits: {all_extracted_digits}")

        return jsonify(meterReading=all_extracted_digits, meterId=meter_id)

      except Exception as e:
        # Log the exception for debugging purposes
        print(f"Exception: {e}")

        # Return an error response
        return jsonify(error="Internal Server Error"), 500

  def preprocessing1(img):
      blur = cv2.GaussianBlur(img, (5, 5), 5)
      gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
      simplethresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
      return simplethresh

  def preprocessing2(img):
      blur = cv2.medianBlur(img, 5)
      gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
      edged = cv2.Canny(gray, 30, 200)
      return edged

  def preprocessing3(img):
      blur = cv2.GaussianBlur(img,(5,5),5)
      gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
      adaptivethreshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
      return adaptivethreshold

  def preprocessing4(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

  def detectwhitearea(img, simplethresh):
      cnts = cv2.findContours(simplethresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      min_area = 9000
      white_dots = []
      for c in cnts:
          area = cv2.contourArea(c)
          if area > min_area:
              # Get the bounding box for the white object
              x, y, w, h = cv2.boundingRect(c)

              # Draw a bounding box around the white object on the original frame
              cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 1)

              # Crop and save the white object as an image
              white_object = img[y:y + h, x:x + w]

              # Return the cropped white object
              return white_object

  def seperate(white_object):
      height, width, _ = white_object.shape
      upper_half = white_object[0:height//2, :]
      lower_half = white_object[height//2:, :]
      return upper_half, lower_half

  def detectblackarea(upper_half,edged):
      contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      largest_contour = None
      largest_contour_area = 0
      # Loop through the contours and find the largest one
      for contour in contours:
          area = cv2.contourArea(contour)
          if area > largest_contour_area:
              largest_contour_area = area
              largest_contour = contour

      x, y, w, h = cv2.boundingRect(largest_contour)
      cropped_region = upper_half[y:y+h-10, x:x+w+10]
      desired_width = 400 # Change to your desired width
      desired_height = 200 # Change to your desired height
      resized_cropped_region = cv2.resize(cropped_region, (desired_width, desired_height))

      return resized_cropped_region


  def OCR(result_black_object, adaptivethreshold):
      contours, _ = cv2.findContours(adaptivethreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Initialize a string to store the extracted digits
      all_extracted_digits = ""

      # Iterate through contours
      for contour in contours:
      # Get the bounding box of the contour
          x, y, w, h = cv2.boundingRect(contour)

      # Set a threshold for the minimum size of the bounding box to filter out small noise
          min_size_threshold = 10
          if w > min_size_threshold and h > min_size_threshold:
              # Extract the region of interest (ROI) from the original image based on the bounding box
              digit_roi = result_black_object[y:y+h, x:x+w]

              # Extract text from the ROI
              extracted_text = pytesseract.image_to_string(digit_roi, config='--psm 8')  # psm 8 for treating the image as a single digit

              # Use regex to keep only digits
              extracted_digits = re.sub(r'\D', '', extracted_text)

              # Append the extracted digits to the overall result
              if len(extracted_digits) == 4:
            # Append the extracted digits to the overall result
                all_extracted_digits += extracted_digits
      return all_extracted_digits


  def detect_qr_codes(lower_half,simplethresh):

      # Decode the barcodes and QR codes
      codes = decode(simplethresh)
      code_data = None

      # Loop through the detected codes (both barcodes and QR codes)
      for code in codes:
          code_data = code.data.decode('utf-8')
          code_type = code.type
          rect_points = code.polygon
          if len(rect_points) == 4:
              cv2.polylines(lower_half, [np.array(rect_points)], True, (0, 255, 0), 2)

          text_x = code.rect.left
          text_y = code.rect.top - 10
          cv2.putText(lower_half, f'{code_type}: {code_data}', (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

          # Print information about the detected code
      # Save the image with detected barcodes and QR codes
      return code_data


  def detectbarcodes(lower_half,gray):
      # Load the image
      # Using pyzbar for detecting QR codes and various barcode types
      codes = decode(gray)
      code_data = None

      # Loop through the detected codes (both barcodes and QR codes)
      for code in codes:
          code_data = code.data.decode('utf-8')
          code_type = code.type
          rect_points = code.polygon
          if len(rect_points) == 4:
              cv2.polylines(lower_half, [np.array(rect_points)], True, (0, 255, 0), 2)

          text_x = code.rect.left
          text_y = code.rect.top - 10
          cv2.putText(lower_half, f'{code_type}: {code_data}', (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

          # Print information about the detected code
          return code_data

except Exception as e:
    print(f"Error setting up ngrok: {e}")

app.run()