import numpy as np
import cv2
from google.colab.patches import cv2_imshow

class VideoEditor:
	def __init__(self):
		pass

	def process(self, f, in_path, out_path):
		stream = cv2.VideoCapture(in_path)
		fourcc = cv2.cv.CV_FOURCC(*'XVID')
		out = cv2.VideoWriter(out_path, fourcc, 20.0, (640,480))

		while 1:
			ret, img = stream.read()
			if ret:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = f(img)
				out.write(img)
				cv2_imshow(img)
			else:
				break




















