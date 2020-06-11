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
import datetime
import imutils
import time
import dlib
import cv2
from pyzbar import pyzbar
import datetime
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

class FacialLandmarks_node:
	def __init__(self):
		# Initializing your ROS Node
		rospy.init_node('FacialLandmarks_node', anonymous=True)

		rospy.on_shutdown(self.shutdown)

		self.rospack = rospkg.RosPack()
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.dlib_filename = self.libraryDir + "/shape_predictor_68_face_landmarks.dat"

		# Give the OpenCV display window a name
		self.cv_window_name = "Facial Landmarks"

		# Create the cv_bridge object
		self.bridge = CvBridge()

		# initialize dlib's face detector (HOG-based) and then create 
		# the facial landmark predictor
		rospy.loginfo("Loading facial landmark predictor...")
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.dlib_filename)

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

		# Get the scan-ed data
		self.getCameraInfo()

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
			cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# show the face number
			cv2.putText(self.cv_image, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				cv2.circle(self.cv_image, (x, y), 1, (0, 0, 255), -1)

	# Preview
	def getFacialLandmark(self):
		while not rospy.is_shutdown():
			try:
				# grab the frame from the threaded video stream, 
				# resize it to have a maximum width of 400 
				# pixels, and convert it to grayscale
				self.cvtImage()

				self.detectFacialLandmark()

				# Overlay some text onto the image display
				self.textInfo()

				# Refresh the image on the screen
				self.dispImage()

			except CvBridgeError as e:
				print(e)

def main(args):
	vn = FacialLandmarks_node()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		rospy.loginfo("[INFO] FacialLandmarks_node [OFFLINE]...")

	cv2.destroyAllWindows()

if __name__ == '__main__':
	rospy.loginfo("[INFO] FacialLandmarks_node [ONLINE]...")
	main(sys.argv)
