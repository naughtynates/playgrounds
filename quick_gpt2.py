import gpt_2_simple as gpt2
import requests
import os
from shutil import make_archive, copyfile
from zipfile import ZipFile
from google.colab import files
from .utils import mount_drive

class GPT2:
	def __init__(self, project_name, base_model):
		self.name = project_name
		self.base_model = base_model
		if not os.path.exists('checkpoint'):
			os.mkdir('checkpoint')
		self.drive_path = '/content/drive/My Drive/'

	def check_model(self):
		base_model = self.base_model
		if not os.path.isdir(os.path.join("models", base_model)):
			gpt2.download_gpt2(model_name=base_model)

	def get_data(self, url):
		with open('data.txt', 'w') as f:
			r = requests.get(url)
			f.write(str(r.content))

	def finetune(self, url, steps):
		self.check_model()
		self.get_data(url)
		base_model = self.base_model
		sess = gpt2.start_tf_sess()
		gpt2.finetune(
			sess,
			'data.txt',
			model_name=self.base_model,
			run_name=self.name,
			steps=steps,
		)

	def save_to_drive(self, weights_name):
		self.drive_path = mount_drive()
		make_archive(weights_name, 'zip', 'checkpoint/' + self.name)
		copyfile(weights_name + '.zip', self.drive_path + weights_name + '.zip')
		os.remove(weights_name + '.zip')

	def load_from_drive(self, weights_name):
		self.drive_path = mount_drive()
		copyfile(self.drive_path + weights_name + '.zip', weights_name + '.zip')
		with ZipFile(weights_name + '.zip', 'r') as z:
			z.extractall('checkpoint/' + self.name)
		os.remove(weights_name + '.zip')
		


	def generate(self, prefix=None, length=1023):
		base_model = self.base_model
		name = self.name
		self.check_model()
		args = dict(
				model_name=base_model, 
				length=length,
				return_as_list=True,
		)
		if prefix is not None:
			args['prefix'] = prefix

		sess = gpt2.start_tf_sess()
		if not os.path.exists('checkpoint/' + name):
			gpt2.load_gpt2(sess, model_name=base_model)
		else:
			args['run_name'] = name
			gpt2.load_gpt2(sess, model_name=base_model, run_name=name)
		return gpt2.generate(sess, **args)[0]


