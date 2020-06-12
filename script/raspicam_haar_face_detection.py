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

from sensor_msgs.msg import RegionOfInterest

class HaarFaceDetector:

	def __init__(self):

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()
		self.roi = RegionOfInterest()
		self.face_detected = False

		rospy.on_shutdown(self.shutdown)

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		# Import haarCascade files
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.haar_filename = self.libraryDir + "/haarcascade_frontalface_default.xml"

		# Path to input Haar cascade for face detection
		self.faceCascade = cv2.CascadeClassifier(self.haar_filename)

		# Subscribe to Image msg
		img_topic = "/raspicam_node_robot/image/compressed"
		self.image_sub = rospy.Subscriber(img_topic, CompressedImage, self.cbImage)

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

		# Clone the original image for displaying purpose later
		self.frameClone = self.cv_image.copy()

		# Put an Info
#		self.putInfo()

		# Detect and Draw Face
#		self.detectHaarFace()

		# Show an Image
		self.showImage()

#		self.pubRegionofInterest()

#		self.take_photo()

	def showImage(self):

		cv2.imshow("Haar Face Detector", self.cv_image)
		cv2.waitKey(1)

	def putInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

#		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(self.cv_image, "{}".format(self.timestr), (10, 15), 
			fontFace, fontScale-0.1, color, thickness, lineType, 
			bottomLeftOrigin)

#		# Clone the original image for displaying purpose later
#		self.frameClone = self.cv_image.copy()

	def shutdown(self):
		try:
			rospy.logwarn("HaarFaceDetector node [OFFLINE]")
		finally:
			cv2.destroyAllWindows()

	def detectHaarFace(self):
		# Create an empty arrays for save rects value later
		self.rects = []
		
		# Detect all faces in the input frame
		self.faceRects = self.faceCascade.detectMultiScale(self.cv_image,
			scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE)

		# Loop over the face bounding boxes
		for (self.fX, self.fY, self.fW, self.fH) in self.faceRects:
			# Extract the face ROI and update the list of bounding boxes
			faceROI = self.cv_image[self.fY:self.fY + self.fH, self.fX:self.fX + self.fW]
			self.rects.append((self.fX, self.fY, self.fX + self.fW, self.fY + self.fH))

			cv2.rectangle(self.cv_image, (self.fX, self.fY), 
				(self.fX + self.fW, self.fY + self.fH), (0, 255, 0), 2)

	def pubRegionofInterest(self):
		# Publish to RegionOfInterest msg
		if len(self.faceRects) != 0:
#			self.roi.x_offset = self.fX
#			self.roi.y_offset = self.fY
#			self.roi.width = self.fX + self.fW
#			self.roi.height = self.fY + self.fH

			self.face_detected = True
		else:
#			self.roi.x_offset = 0
#			self.roi.y_offset = 0
#			self.roi.width = 0
#			self.roi.height = 0

			self.face_detected = False

#		self.roi_pub.publish(self.roi)

	def take_photo(self):
		img_title = self.timestr + "-photo.jpg"
		if self.face_detected:
			cv2.imwrite(img_title, self.frameClone)
#			rospy.logwarn("Face Detect")
			rospy.sleep(1)
		else:
#			rospy.logwarn("No Face Detect")
			pass

def main(args):
	face = HaarFaceDetector()
	rospy.init_node("haarcascade_frontalface", anonymous=False)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
