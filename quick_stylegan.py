from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
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
		self.people[name] = {
			'images': filenames, 
			'face': face,
			'embedding': embedding,
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
		def processor(img):
			cv2.imwrite('temp.jpg', img)
			try:
				face, img = self.image_swap('temp.jpg', face_map['all'])
			except:
				img = cv2.resize(img, (768,432))
			clear_output()
			plt.imshow(img)
			plt.pause(0.000000001)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
		if autosave:
			auth.authenticate_user()
		editor = VideoEditor(processor)
		editor.process(self.files[filename], out_path)
		if autosave:
			save_to_drive(out_path, out_path)


















