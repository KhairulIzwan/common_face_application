#!/usr/bin/env python

#Title: Python Subscriber for Tank Navigation
#Author: Khairul Izwan Bin Kamsani - [23-01-2020]
#Description: Tank Navigation Subcriber Nodes (Python)

from __future__ import print_function
from __future__ import division

# import the necessary packages
from imutils import face_utils
import imutils
import time
import cv2
import os
import rospkg
import sys
import rospy
import numpy as np

# import the necessary ROS messages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

class HaarFaceDetector:

	def __init__(self):

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()
		self.image_received = False

		rospy.on_shutdown(self.shutdown)

		# Import haarCascade files
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.haar_filename = self.libraryDir + "/haarcascade_frontalface_default.xml"

		# Path to input Haar cascade for face detection
		self.faceCascade = cv2.CascadeClassifier(self.haar_filename)

		# Subscribe to Image msg
		self.img_topic = "/raspicam_node_robot/image/compressed"
		self.image_sub = rospy.Subscriber(self.img_topic, CompressedImage, self.cbImage)

		rospy.logwarn("HaarFaceDetector Node [ONLINE]...")

		# Allow up to one second to connection
		rospy.sleep(1)

	# Get the width and height of the image
	def cbCameraInfo(self):

		self.imgWidth = rospy.get_param("/raspicam_node_robot/width") 
		self.imgHeight = rospy.get_param("/raspicam_node_robot/height") 

		rospy.set_param("/raspicam_node_robot/vFlip", True)

	def cbImage(self, msg):

		# Convert image to OpenCV format
		try:
			self.cv_image = np.fromstring(msg.data, np.uint8)
			self.cv_image = cv2.imdecode(self.cv_image, cv2.IMREAD_COLOR)

			# OPTIONAL -- image-rotate """
			self.cv_image = imutils.rotate(self.cv_image, angle=-90)
			self.cv_image = cv2.flip(self.cv_image,1)
		except CvBridgeError as e:
			print(e)

		self.image_received = True
		self.image = self.cv_image

	def preview(self):

		cv2.imshow("Haar Face Detector", self.image)
		cv2.waitKey(1)

	# Overlay some text onto the image display
	def showInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(self.image, "{}".format(self.timestr), (10, 20), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.image, "Sample", (10, self.imgHeight-10), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
			color, thickness, lineType, bottomLeftOrigin)

	def shutdown(self):
		rospy.logwarn("HaarFaceDetector node [OFFLINE]")
		cv2.destroyAllWindows()

	def detectHaarFace(self):
		# Create an empty arrays for save rects value later
#		self.rects = []
		
		# Detect all faces in the input frame
		self.faceRects = self.faceCascade.detectMultiScale(self.cv_image,
			scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE)

		# Loop over the face bounding boxes
		for (self.fX, self.fY, self.fW, self.fH) in self.faceRects:
			# Extract the face ROI and update the list of bounding boxes
			faceROI = self.image[self.fY:self.fY + self.fH, self.fX:self.fX + self.fW]
#			self.rects.append((self.fX, self.fY, self.fX + self.fW, self.fY + self.fH))

			cv2.rectangle(self.image, (self.fX, self.fY), 
				(self.fX + self.fW, self.fY + self.fH), (0, 255, 0), 2)

		self.cbFaceDetected()

	def cbFaceDetected(self):

		if len(self.faceRects) != 0:
			self.face_detected = True
		else:
			self.face_detected = False

	def take_photo(self):

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")
		img_title = self.timestr + "-photo.png"
		if self.image_received:
#			self.preview()
			self.detectHaarFace()
			if self.face_detected:
				cv2.imwrite(img_title, self.image)
				# Sleep to give the last log messages time to be sent
				rospy.logerr("Image Captured")
			else:
				pass
		else:
			pass

def main(args):

	rospy.init_node("haarcascade_frontalface", anonymous=False)
	face = HaarFaceDetector()

	while not rospy.is_shutdown():
		face.take_photo()

if __name__ == '__main__':

	main(sys.argv)
