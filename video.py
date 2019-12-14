import numpy as np
import cv2
from google.colab.patches import cv2_imshow

class VideoEditor:
	def __init__(self):
		pass

	def process(self, f, in_path, out_path):
		stream = cv2.VideoCapture(in_path)
		#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		#out = cv2.VideoWriter(out_path, fourcc, 20.0, (640,480))
		out = cv2.VideoWriter(out_path, -1, 20.0, (640,480))
		count = 0
		while 1:
			ret, img = stream.read()
			count += 1
			print('Frame:', count)
			if not ret or cv2.waitKey(25) & 0XFF == ord('q'):
				cv2.destroyAllWindows()
				break
			else:
				img = f(img)
				out.write(img)




















