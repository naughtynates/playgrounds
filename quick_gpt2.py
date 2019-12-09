import gpt_2_simple as gpt2
import requests
import os
from google.colab import files
import shutil
from .utils import save_to_drive

class GPT2:
	def __init__(self, name, base_model):
		self.name = name
		self.base_model = base_model
		if not os.path.exists('checkpoint'):
			os.mkdir('checkpoint')

	def check_model(self):
		base_model = self.base_model
		if not os.path.isdir(os.path.join("models", base_model)):
			gpt2.download_gpt2(model_name=base_model)

	def get_data(self, url):
		with open('data.txt', 'w') as f:
			r = requests.get(url)
			f.write(str(r.content))

	def finetune(self, url, steps):
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

	def download(self):
		name = self.name
		shutil.make_archive(name, 'zip', 'checkpoint/' + name)
		save_to_drive(name + '.zip', name + '.zip')
		print()
		os.remove(name + '.zip')

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


