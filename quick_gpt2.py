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
		self.drive_path = '/content/drive/My Drive/'
		if not os.path.exists('checkpoint'):
			os.mkdir('checkpoint')
		self.check_model()

	def check_model(self):
		if not os.path.isdir(os.path.join("models", self.base_model)):
			gpt2.download_gpt2(model_name=self.base_model)

	def get_data(self, url):
		with open('data.txt', 'wb') as f:
			r = requests.get(url)
			f.write(r.content)

	def finetune(self, url, steps):
		self.get_data(url)
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
		self.unpack_weights(weights_name + '.zip')

	def load(self, bot_name=None, url=None):
		urls = {
			'shakespeare': 'https://drive.google.com/open?id=1zvfzFcT2YsJXHNLuAmClnuMHoJuTNWyd',
		}
		if bot_name is not None:
			url = urls[bot_name]
		print('downloading weights...')
		os.system('wget ' + url)
		self.unpack_weights(url.split('/')[-1])


	def unpack_weights(self, filename):
		with ZipFile(filename, 'r') as z:
			z.extractall('checkpoint/' + self.name)
		os.remove(filename)

	def generate(self, prefix=None, num_samples=1, length=1023):
		self.check_model()
		args = dict(
				model_name=self.base_model, 
				length=length,
				return_as_list=True,
				num_samples=num_samples,
		)
		if prefix is not None:
			args['prefix'] = prefix

		sess = gpt2.start_tf_sess()
		if not os.path.exists('checkpoint/' + self.name):
			gpt2.load_gpt2(sess, model_name=self.base_model)
		else:
			args['run_name'] = self.name
			gpt2.load_gpt2(sess, run_name=self.name)
		output = [str(x) for x in gpt2.generate(sess, **args)]
		if num_samples == 1:
			return output[0]
		else:
			return output

