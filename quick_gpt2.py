import gpt_2_simple as gpt2
import os

class GPT2:
	def __init__(self, base_model, custom=None):
		self.base_model = base_model
		self.custom = custom


	def check_model(self):
		base_model = self.base_model
		if not os.path.isdir(os.path.join("models", base_model)):
			gpt2.download_gpt2(model_name=base_model)

	def generate(self, length=1023, prefix=None):
		self.check_model()
		sess = gpt2.start_tf_sess()
		gpt2.load_gpt2(sess, model_name=base_model)
		args = dict(
				model_name=base_model, 
				length=length,
				return_as_list=True,
		)
		if prefix is not None:
			args['prefix'] = prefix
		return gpt2.generate(sess, **args)[0]

	def finetune(self):
		pass


