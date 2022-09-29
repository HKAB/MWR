from flask import Flask, render_template, redirect, url_for, request, jsonify
app = Flask(__name__, template_folder='templates')
import cv2 as cv2
import numpy as np
import urllib.request
from PIL import Image
import io

@app.route('/')
def home():
   return render_template('home.html')




@app.route('/get-age', methods=['POST'])
def age():
   names = "Đạt"
   img = request.data
   print(type(img))
   print("haha")
   # img = img.decode("utf-8")
   image = Image.open(io.BytesIO(img))

   open_cv_image = np.array(image) 
   # Convert RGB to BGR 
   open_cv_image = open_cv_image[:, :, ::-1].copy() 

   image1 = Image.fromarray(open_cv_image)
   print(type(image1))
   # print(image.size)

   imgByteArr = io.BytesIO()
  # image.save expects a file as a argument, passing a bytes io ins
   image1.save(imgByteArr, format=image.format)
   # Turn the BytesIO object back into a bytes object
   imgByteArr = imgByteArr.getvalue()
   print(type(imgByteArr))

   return imgByteArr
 
if __name__ == '__main__':
   app.run(debug=True) 

   