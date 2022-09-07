import streamlit as st
from base64 import decodestring
import pandas as pd
import tempfile
import time 
import glob
from io import BytesIO
import base64
import io
import cv2
import os
import numpy as np
import imutils
import sys
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


letters = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]


le = LabelEncoder()
z=le.fit_transform(letters)

### Excluding Imports ###
st.title("CAPTCHA EXTRACTION")
def file_selector(folder_path='data/predict'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# if uploaded_file is not None:
# 	image = Image.open(uploaded_file)
# 	st.image(image, caption='Uploaded Image.', use_column_width=True)
# 	st.write("")
# 	st.write("Classifying...")
# 	print(uploaded_file)
st.image(filename,caption='Uploaded Image.', use_column_width=True)
img_cv = cv2.imread(filename)
#############################################################################
# converting to black and white images 
# submit_button = st.form_submit_button(label='Submit')
if st.button("extract"):
	hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
	lower_white = np.array([0, 0, 212], dtype=np.uint8)
	upper_white = np.array([131, 255, 255], dtype=np.uint8)
	mask = cv2.inRange(hsv, lower_white, upper_white)
	kernel = np.ones((1,1),np.uint8)
	img_e = cv2.erode(mask,kernel,iterations=2)
	img_e=255-img_e
	Capatcha_dir_1 = tempfile.mkdtemp()
	cv2.imwrite(Capatcha_dir_1+"ba.jpg",img_e)
	# print(img_e.shape)
	####################################################################
	# croping to each letters and finally stored in to_model(variable)
	img = cv2.imread(Capatcha_dir_1+"ba.jpg")
	# print(img.shape)
	rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
	black = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
	# print(black.shape)
	kernel = np.ones((1,1), np.uint8)
	dilate = cv2.erode(black,kernel,iterations = 1)
	erode = cv2.dilate(dilate,kernel,iterations = 2)
	ret,thresh = cv2.threshold(erode,1,255,cv2.THRESH_BINARY_INV)
	img_dilation = cv2.erode(thresh,kernel,iterations = 2)
	ctrs = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(ctrs)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = False)[:10]
	(cnts, boundingBoxes) = sort_contours(cnts, method="left-to-right")
	roi_images = []
	for i, ctr in enumerate(cnts):
	    # Get bounding box
	    x, y, w, h = cv2.boundingRect(ctr)
	#     print(x,y,w,h)
	    if w>= 6 and h>=9 and w<=20:

	        # Getting ROI
	        roi = img[y:y+h, x:x+w]
	#         roi =cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
	        
	        roi = cv2.resize(roi,(15,10))
	#         roi = roi/255
	#         roi =cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
	#         roi.reshape(1,-1)
	#         roi = np.reshape(roi, (15,10,1))
	        # print(roi.shape)
	        roi_images.append(roi)
	        cv2.imwrite("data/abc/"+str(i)+".jpg",roi)

	to_model =   np.array(roi_images)
	# to_model = np.reshape(to_model,(6, 10, 15,1))

	# print(to_model.shape)
	if to_model.shape[0] != 6:
	    sys.exit()


	# to_model = extract()
	le = LabelEncoder()
	le.fit_transform(letters)
	model = load_model("data/model2_10_15_3.h5")
	pre = model.predict(to_model)
	classes=np.argmax(pre,axis=1)
	list_value = le.inverse_transform(classes)
	captcha_value = "".join(list_value)

	st.write(captcha_value)
