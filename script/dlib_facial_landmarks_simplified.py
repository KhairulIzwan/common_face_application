#!/usr/bin/env python

################################################################################
## {Description}: 
## Face Alignment with OpenCV and Python with ROS
################################################################################
## Author: Khairul Izwan Bin Kamsani
## Version: {1}.{0}.{0}
## Email: {wansnap@gmail.com}
################################################################################
## {Disclaimer}: 
## This code were revised from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
################################################################################

from __future__ import print_function
from __future__ import division

# import the necessary packages
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import os
import rospkg
import sys
import rospy

# import the necessary ROS messages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

from sensor_msgs.msg import RegionOfInterest

class DlibFacialLandmarks:
	def __init__(self):

		self.bridge = CvBridge()
		self.rospack = rospkg.RosPack()
		self.roi = RegionOfInterest()

		rospy.on_shutdown(self.shutdown)

		# initialize dlib's face detector (HOG-based) and then create 
		# the facial landmark predictor
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.dlib_filename = self.libraryDir + "/shape_predictor_68_face_landmarks.dat"

		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.dlib_filename)

		# Subscribe to Image msg
		image_topic = "/cv_camera/image_raw"
		self.image_sub = rospy.Subscriber(image_topic, Image, self.cbImage)

		# Subscribe to CameraInfo msg
		cameraInfo_topic = "/cv_camera/camera_info"
		self.cameraInfo_sub = rospy.Subscriber(cameraInfo_topic, CameraInfo, 
			self.cbCameraInfo)

		# Publish to RegionOfInterest msgs
		roi_topic = "/faceROI"
		self.roi_pub = rospy.Publisher(roi_topic, RegionOfInterest, queue_size=10)

	def cbCameraInfo(self, msg):
		# Get CameraInfo
		try:
			self.imgWidth = msg.width
			self.imgHeight = msg.height
		except CvBridgeError as e:
			print(e)

	def cbImage(self, msg):
		# Convert image to OpenCV format
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		# Put an Info
		self.putInfo()

		# Detect and Draw Facial Landmarks
		self.detectFacialLandmark()

		# Show an Image
		self.showImage()

	def showImage(self):

		cv2.imshow("Dlib Facial Landmarks", self.cv_image)
		cv2.waitKey(1)

	def putInfo(self):

		fontFace = cv2.FONT_HERSHEY_DUPLEX
		fontScale = 0.5
		color = (255, 255, 255)
		thickness = 1
		lineType = cv2.LINE_AA
		bottomLeftOrigin = False # if True (text upside down)

		timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(self.cv_image, "{}".format(timestr), (10, 15), 
			fontFace, fontScale-0.1, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.cv_image, "Sample", (10, self.imgHeight-10), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
			color, thickness, lineType, bottomLeftOrigin)

	def shutdown(self):
		try:
			rospy.logwarn("DlibFacialLandmarks node [OFFLINE]...")

		finally:
			cv2.destroyAllWindows()

	def detectFacialLandmark(self):
		gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = self.detector(gray, 0)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), 
				(0, 255, 0), 2)

			# Publish to RegionOfInterest msg
			self.roi.x_offset = x
			self.roi.y_offset = y
			self.roi.width = x + w
			self.roi.height = y + h

			self.roi_pub.publish(self.roi)

			# show the face number
			cv2.putText(self.cv_image, "Face #{}".format(i + 1), 
				(x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
				(0, 255, 0), 2)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(self.cv_image, (x, y), 1, (0, 0, 255), -1)

def main(args):
	face = DlibFacialLandmarks()
	rospy.init_node("dlib_facial_landmarks", anonymous=False)

	try:
		rospy.spin()
	except KeyboardInterrupt:
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
