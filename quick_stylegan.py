from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
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

	def add_person(self, name):
		filenames = [k for k,v in files.upload().items()]
		if name in self.people:
			self.people[name] += filenames
		else:
			self.people[name] = filenames

	def add_file(self, name):
		filename = [k for k,v in files.upload().items()][0]
		self.files[name] = filename

	def img_swap(self, filename, person):
		src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(filename, self.fd, self.fp, self.idet)
		tar, emb_tar = utils.get_tar_inputs(self.people[person], self.fd, self.fv)
		out = self.model.inference(src, mask, tar, emb_tar)
		face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
		img = utils.post_process_result(filename, self.fd, face, aligned_im, src, x0, y0, x1, y1, landmarks)
		return face, img

	def video_swap(self, filename, out_path, id_map={}, autosave=False):
		def processor(img):
			cv2.imwrite('temp.jpg', img)
			face, img = self.img_swap('temp.jpg', 'yaka')
			clear_output()
			plt.imshow(img)
			plt.pause(0.000000001)
			return img
		if autosave:
			auth.authenticate_user()
		editor = VideoEditor()
		editor.process(processor, self.files[filename], out_path)
		if autosave:
			save_to_drive(out_path, out_path)


















