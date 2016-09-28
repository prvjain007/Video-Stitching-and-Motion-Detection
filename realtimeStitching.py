from __future__ import print_function
from a.basicmotiondetector import BasicMotionDetector
from a.panorama import Stitcher
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

# Initialize the video streams
leftStream = VideoStream(src = 1).start()
rightStream = VideoStream(src = 2).start()

# Initialize the image stitcher, motion detector, and total number of frames read
stitcher = Stitcher()
motion = BasicMotionDetector(minArea = 500)
total = 0

# Loop over all the image streams
while True:
	# Read the frames from the given streams
	left = leftStream.read()
	right = rightStream.read()

	# Resize the input so that the input images are not elliptical
	left = imutils.resize(left, width = 400)
	right = imutils.resize(right, width = 400)

	# Stitch the frames together to form the panorama
	# Select frames from left to right
	result = stitcher.stitch([left, right])

	# If homography computation failed then break
	if result is None:
		print("failed to compute homography matrix")
		break

	# Convert the stitched output to grayscale and blur it slightly to increase accuracy
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	locs = motion.update(gray)

	# Only find the output if a nice average has been found
	if total > 32 and len(locs) > 0:
		# As clustering is not used to find the object, we use global object boxes
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

                # Accumulate the object boundaries	
		for l in locs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))

		# Draw the bounding box
		cv2.rectangle(result, (minX, minY), (maxX, maxY),
			(0, 0, 255), 3)

	# Show the output images
	cv2.imshow("Result", result)
	cv2.imshow("Left Frame", left)
	cv2.imshow("Right Frame", right)
	key = cv2.waitKey(1) & 0xFF

	# If the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
print("DONE!")
