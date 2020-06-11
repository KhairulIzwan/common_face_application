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

# import the necessary ROS messages
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

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

		# Import haarCascade files
		self.p = os.path.sep.join([self.rospack.get_path('common_face_application')])
		self.libraryDir = os.path.join(self.p, "library")

		self.haar_filename = self.libraryDir + "/haarcascade_frontalface_default.xml"

		# Path to input Haar cascade for face detection
		self.faceCascade = cv2.CascadeClassifier(self.haar_filename)

		# Subscribe to Image msg
		image_topic = "/cv_camera/image_raw"
		# Un-comment if is using raspicam
#		img_topic = "/raspicam_node_robot/image/compressed"
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

		# Detect and Draw Face
		self.detectHaarFace()

		# Show an Image
		self.showImage()

		self.pubRegionofInterest()

		self.take_photo()

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

		self.timestr = time.strftime("%Y%m%d-%H:%M:%S")

		cv2.putText(self.cv_image, "{}".format(self.timestr), (10, 15), 
			fontFace, fontScale-0.1, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.cv_image, "Sample", (10, self.imgHeight-10), 
			fontFace, fontScale, color, thickness, lineType, 
			bottomLeftOrigin)
		cv2.putText(self.cv_image, "(%d, %d)" % (self.imgWidth, self.imgHeight), 
			(self.imgWidth-100, self.imgHeight-10), fontFace, fontScale, 
			color, thickness, lineType, bottomLeftOrigin)

		# Clone the original image for displaying purpose later
		self.frameClone = self.cv_image.copy()

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
			self.roi.x_offset = self.fX
			self.roi.y_offset = self.fY
			self.roi.width = self.fX + self.fW
			self.roi.height = self.fY + self.fH

			self.face_detected = True
		else:
			self.roi.x_offset = 0
			self.roi.y_offset = 0
			self.roi.width = 0
			self.roi.height = 0

			self.face_detected = False

		self.roi_pub.publish(self.roi)

	def take_photo(self):
		img_title = self.timestr + "-photo.jpg"
		if self.face_detected:
			cv2.imwrite(img_title, self.frameClone)
		else:
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
