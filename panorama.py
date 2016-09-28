# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
	def __init__(self):
		# Initialize the homography cache
		self.isv3 = imutils.is_cv3()
		self.cachedH = None

	def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		(imageB, imageA) = images

		# If homography matrix is not computed then compute it and cache it
		if self.cachedH is None:
			# detect keypoints and extract
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)

			# Match the features
			M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

			if M is None:
				return None

			# Create the cache
			self.cachedH = M[1]

		# Apply a perspective transform to stitch the images together using the cached homography matrix
		result = cv2.warpPerspective(imageA, self.cachedH, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

		return result

	def detectAndDescribe(self, image):
		# Convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect and extract features
		if self.isv3:
			# Detect features
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
		else:
                        detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# Convert the keypoints from KeyPoint objects to NumPy arrays
		kps = np.float32([kp.pt for kp in kps])

		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# Compute the raw matches and initialize the list of actual matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		for m in rawMatches:
			# Lowe's Ratio Test
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# Homography cannot be computed if there are less than 4 points
		if len(matches) > 4:
			# Construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# Compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

			return (matches, H, status)
		return None
