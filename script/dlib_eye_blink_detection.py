#!/usr/bin/env python

################################################################################
## {Description}: 
## Eye blink detection with OpenCV, Python, and dlib with ROS
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################
## {Disclaimer}: 
## This code were revised from https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
################################################################################

from __future__ import print_function
from __future__ import division

# import the necessary packages
from imutils import face_utils
from pyzbar import pyzbar
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import datetime
import imutils
import time
import dlib
import cv2
import os
import sys
import numpy as np

import rospy
import rospkg

# import the necessary ROS messages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

class EyeBlinkDetection_node:
	def __init__(self):
		# Initializing your ROS Node
		rospy.init_node('EyeBlinkDetection_node', anonymous=True)

		rospy.on_shutdown(self.shutdown)

		self.rospack = rospkg.RosPack()
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.dlib_filename = self.libraryDir + "/shape_predictor_68_face_landmarks.dat"

		# Give the OpenCV display window a name
		self.cv_window_name = "Facial Landmarks"

		# Create the cv_bridge object
		self.bridge = CvBridge()

		# define two constants, one for the eye aspect ratio to indicate
		# blink and then a second constant for the number of consecutive
		# frames the eye must be below the threshold
		self.EYE_AR_THRESH = 0.3
		self.EYE_AR_CONSEC_FRAMES = 3

		# initialize the frame counters and the total number of blinks
		self.COUNTER = 0
		self.TOTAL = 0

		# initialize dlib's face detector (HOG-based) and then create 
		# the facial landmark predictor
		rospy.loginfo("Loading facial landmark predictor...")
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.dlib_filename)

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# Subscribe to the raw camera image topic
		self.imgRaw_sub = rospy.Subscriber("/cv_camera/image_raw", Image)

		# Subscribe to the camera info topic
		self.imgInfo_sub = rospy.Subscriber("/cv_camera/camera_info", CameraInfo)

		self.getFacialLandmark()

	def getImage(self):
		# Wait for the topic
		self.image = rospy.wait_for_message("/cv_camera/image_raw", Image)

	# Get the width and height of the image
	def getCameraInfo(self):
		# Wait for the topic
		self.camerainfo = rospy.wait_for_message("/cv_camera/camera_info", CameraInfo)

	# Overlay some text onto the image display
	def textInfo(self):
		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		cv2.putText(self.cv_image, "Sample", (10, self.camerainfo.height-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
		cv2.putText(self.cv_image, "(%d, %d)" % (self.camerainfo.width, self.camerainfo.height), (self.camerainfo.width-100, self.camerainfo.height-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA, False)

	# Refresh the image on the screen
	def dispImage(self):
		cv2.imshow(self.cv_window_name, self.cv_image)
		cv2.waitKey(1)

	# Shutdown
	def shutdown(self):
		try:
			rospy.logwarn("CameraPreview_node [OFFLINE]...")

		finally:
			cv2.destroyAllWindows()

	# grab the frame from the threaded video stream, resize it to have a 
	# maximum width of 400 pixels, and convert it to grayscale
	def cvtImage(self):
		# Get the scan-ed data
		self.getImage()

		# Convert the raw image to OpenCV format """
		self.cv_image = self.bridge.imgmsg_to_cv2(self.image, "bgr8")

		# TODO:
		#self.cv_image = imutils.rotate(self.cv_image, angle=180)
		self.cv_image_copy = self.cv_image.copy()

	def detectFacialLandmark(self):
		gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = self.detector(gray, 1)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[self.lStart:self.lEnd]
			rightEye = shape[self.rStart:self.rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(self.cv_image, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(self.cv_image, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < self.EYE_AR_THRESH:
				self.COUNTER += 1

			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
					self.TOTAL += 1

				# reset the eye frame counter
				self.COUNTER = 0

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(self.cv_image, "Blinks: {}".format(self.TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(self.cv_image, "EAR: {:.2f}".format(ear), (self.camerainfo.width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# Preview
	def getFacialLandmark(self):
		while not rospy.is_shutdown():
			try:
				# grab the frame from the threaded video stream, 
				# resize it to have a maximum width of 400 
				# pixels, and convert it to grayscale
				self.cvtImage()

				# Get the scan-ed data
				self.getCameraInfo()

				self.detectFacialLandmark()

				# Overlay some text onto the image display
				self.textInfo()

				# Refresh the image on the screen
				self.dispImage()

			except CvBridgeError as e:
				print(e)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def main(args):
	vn = EyeBlinkDetection_node()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("[INFO] EyeBlinkDetection_node [OFFLINE]...")

	cv2.destroyAllWindows()

if __name__ == '__main__':
	rospy.loginfo("[INFO] EyeBlinkDetection_node [ONLINE]...")
	main(sys.argv)
