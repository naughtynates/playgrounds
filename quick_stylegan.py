from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
import numpy as np
from utils import utils
from matplotlib import pyplot as plt
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

class StyleGAN:
	def __init__(self):
		self.model = FaceTranslationGANInferenceModel()
		self.fv = FaceVerifier(classes=512)
		self.fp = face_parser.FaceParser()
		self.fd = face_detector.FaceAlignmentDetector()
		self.idet = IrisDetector()
		self.people = {}
		self.images = {}

	def add_person(self, name):
		filenames = [k for k,v in files.upload().items()]
		if name in self.people:
			self.people[name] += filenames
		else:
			self.people[name] = filenames

	def add_image(self, name):
		filename = [k for k,v in files.upload().items()][0]
		self.images[name] = filename

	def simple_swap(self, image, person):
		src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(image, self.fd, self.fp, self.idet)
		tar, emb_tar = utils.get_tar_inputs(self.people[person], self.fd, self.fv)
		out = self.model.inference(src, mask, tar, emb_tar)
		face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
		img = utils.post_process_result(fn_src, self.fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
		return face, img


