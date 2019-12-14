from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
import face_recognition
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import os
import cv2
from utils import utils
from google.colab import auth
from google.colab import files
import warnings
from .utils import save_to_drive
from .video import VideoEditor

warnings.filterwarnings("ignore")

class StyleGAN:
	def __init__(self):
		self.model = FaceTranslationGANInferenceModel()
		self.fv = FaceVerifier(classes=512)
		self.fp = face_parser.FaceParser()
		self.fd = face_detector.FaceAlignmentDetector()
		self.idet = IrisDetector()
		self.people = {}
		self.files = {}

	def add_person(self, name, paths=[], urls=[]):
		if len(paths) > 0:
			filenames = paths
		elif len(urls) > 0:
			filenames = []
			for url in urls:
				os.system('wget ' + url)
				filenames.append(url.split('/')[-1])
		else:
			filenames = [k for k,v in files.upload().items()]
		face, embedding = utils.get_tar_inputs(filenames, self.fd, self.fv)
		img = cv2.imread(filenames[0])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		bounding_boxes = face_recognition.face_locations(img)
		encs = face_recognition.face_encodings(img, bounding_boxes)
		self.people[name] = {
			'images': filenames, 
			'face': face,
			'embedding': embedding,
			'rec_enc': encs[0]
		}

	def add_file(self, name, url=None, path=None):
		if path is not None:
			filename = path
		else:
			if url is not None:
				os.system('wget ' + url)
				filename = url.split('/')[-1]
			else:
				filename = [k for k,v in files.upload().items()][0]
		self.files[name] = filename

	def image_swap(self, filename, person):
		src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(filename, self.fd, self.fp, self.idet)
		tar, emb_tar = self.people[person]['face'], self.people[person]['embedding']
		out = self.model.inference(src, mask, tar, emb_tar)
		face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
		img = utils.post_process_result(filename, self.fd, face, aligned_im, src, x0, y0, x1, y1, landmarks)
		return face, img

	def video_swap(self, filename, out_path, face_map={}, autosave=False):
		def processor(img, frame_num):
			img = self.auto_resize(img)
			bounding_boxes = face_recognition.face_locations(img)
			if len(bounding_boxes) > 0:
				try:
					try:
						src_encs = face_recognition.face_encodings(img, bounding_boxes)
						tar_encs = [self.people[x]['rec_enc'] for x in face_map.keys()]
						for i in range(len(src_encs)):
							bb = bounding_boxes[i]
							scores = face_recognition.compare_faces(tar_encs, src_encs[i])
							matches = [list(face_map.keys())[i] for i in range(len(scores)) if scores[i] == 1]
							for match in matches:
								adj = 30
								x1, x2 = np.max([0, bb[0] - adj]), np.min([img.shape[0], bb[2] + adj])
								y1, y2 = np.max([0, bb[3] - adj]), np.min([img.shape[1], bb[1] + adj])
								face_img = img[x1:x2, y1:y2]
								cv2.imwrite('temp.jpg', face_img)
								face, face_img = self.image_swap('temp.jpg', face_map[match])
								plt.pause(0.000000001)
								img[x1:x2, y1:y2] = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
					except AssertionError:
						pass
					if frame_num % 30 == 0:
						clear_output()
				except Exception:
					pass
			plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
			plt.pause(0.000000001)
			return img
		if autosave:
			auth.authenticate_user()
		editor = VideoEditor(processor)
		editor.process(self.files[filename], out_path)
		if autosave:
			save_to_drive(out_path, out_path)

	def auto_resize(self, img, max_size=740):
		if np.max(img.shape) > max_size:
			ratio = max_size / np.max(img.shape)
			img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
		return img

















