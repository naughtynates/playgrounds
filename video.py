import numpy as np
import cv2

class VideoEditor:
	def __init__(self, f):
		self.f = f

	def process(self, in_path, out_path):
		stream = cv2.VideoCapture(in_path)
		fps = stream.get(cv2.CAP_PROP_FPS)
		img = None
		while img is None:
			ret, img = stream.read()
			img = self.f(img, 0)
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		out = cv2.VideoWriter(out_path, fourcc, fps, (img.shape[1], img.shape[0]))
		frame_num = 0
		while 1:
			ret, img = stream.read()
			frame_num += 1
			if not ret or cv2.waitKey(25) & 0XFF == ord('q'):
				cv2.destroyAllWindows()
				break
			else:
				img = self.f(img, frame_num)
				out.write(img)




















