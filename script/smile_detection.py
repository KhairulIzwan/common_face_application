#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

# import the necessary ROS packages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import rospy

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import time
import rospkg
import os
import numpy as np

class SmileDetector:

	def __init__(self):

		rospy.logwarn("SmileDetector node [ONLINE]")

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()

		self.image_recieved = False
		self.face_detected = False

		rospy.on_shutdown(self.shutdown)

		## load the face detector cascade and smile detector CNN
		# Import haarCascade files
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.haar = self.libraryDir + "/haarcascade_frontalface_default.xml"

		# Path to input Haar cascade for face detection
		self.faceCascade = cv2.CascadeClassifier(self.haar)

		# Import model files
		self.p = os.path.sep.join([self.rospack.get_path('smile_detection')])
		self.libraryDir = os.path.join(self.p, "model")

		self.model = self.libraryDir + "/lenet.hdf5"

		self.smile = load_model(self.model)

		# Subscribe to Image msg
		image_topic = "/cv_camera/image_raw"
		self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

		# Subscribe to CameraInfo msg
		cameraInfo_topic = "/cv_camera/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo, 
			self.cbCameraInfo)

		rospy.sleep(1)

	# Get CameraInfo
	def cbCameraInfo(self, msg):

		self.imgWidth = msg.width
		self.imgHeight = msg.height

	# Convert image to OpenCV format
	def cbImage(self, msg):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		self.image_recieved = True
		self.image = cv_image

	def showImage(self, winName, img):

		cv2.imshow(winName, img)
		cv2.waitKey(1)

	def putInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.4
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(self.image, "{}".format(self.timestr), (10, 15), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.image, "Sample", (10, self.imgHeight-10), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
			color, thickness, lineType, bottomLeftOrigin)

		# Clone the original image for displaying purpose later
		self.frameClone = self.image.copy()

	def shutdown(self):
		try:
			rospy.logwarn("SmileDetector node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

	def detectHaarFace(self):

		if self.image_recieved:
			# Detect all faces in the input frame
			self.faceRects = self.faceCascade.detectMultiScale(self.image,
				scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30),
				flags = cv2.CASCADE_SCALE_IMAGE)

			# Loop over the face bounding boxes
			for (fX, fY, fW, fH) in self.faceRects:
				# Extract the face ROI and update the list of bounding boxes
				faceROI = self.image[fY:fY + fH, fX:fX + fW]

				# convert it to grayscale
				grayROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

				# resize it to a fixed 28x28 pixels, and then prepare the
				# ROI for classification via the CNN
				roi = cv2.resize(grayROI, (28, 28))
				roi = roi.astype("float") / 255.0
				roi = img_to_array(roi)
				roi = np.expand_dims(roi, axis=0)

				# determine the probabilities of both "smiling" and "not
				# smiling", then set the label accordingly
				(notSmiling, smiling) = self.smile.predict(roi)[0]
				label = "Smiling" if smiling > notSmiling else "Not Smiling"

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(self.image, label, (fX, fY - 10),
					cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)
				cv2.rectangle(self.image, (fX, fY), (fX+fW, fY+fH), 
					(0, 255, 0), 2)

			self.putInfo()
			self.showImage("Haar Face Detector", self.image)


if __name__ == '__main__':

	# Initializing your ROS Node
	rospy.init_node('smile_detector', anonymous=False)
	face = SmileDetector()

	# Camera preview
	while not rospy.is_shutdown():
		face.detectHaarFace()
