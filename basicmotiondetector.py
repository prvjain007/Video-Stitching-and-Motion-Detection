# import the necessary packages
import imutils
import cv2

class BasicMotionDetector:
	def __init__(self, accumWeight = 0.5, deltaThresh = 5, minArea = 5000):
		# Determine the OpenCV version, followed by storing the the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required for "motion" to be reported
		self.isv2 = imutils.is_cv2()
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea

		# Initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		# Initialize the list of locations containing motion
		locs = []

		# If the average image is None, initialize it
		if self.avg is None:
			self.avg = image.astype("float")
			return locs

		# Otherwise, find and accumulate the average (weighted) between consecutive frames
		cv2.accumulateWeighted(image, self.avg, self.accumWeight)

		# Compute the pixelwise difference between the current frame and the accumulated average
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

		# Threshold the delta image and apply a series of dilations to help fill in holes
		thresh = cv2.threshold(frameDelta, self.deltaThresh, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations = 2)

		# Find contours in the thresholded image
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if self.isv2 else cnts[1]

		for c in cnts:
			# Add the contour to the locations list if it exceeds the minimum area
			if cv2.contourArea(c) > self.minArea:
				locs.append(c)
		
		return locs
