import gpt_2_simple as gpt2
import os

def check_model(c):
  if not os.path.isdir(os.path.join("models", c['model_name'])):
	  gpt2.download_gpt2(model_name=c['model_name'])

def generate(c, prefix=None):
  check_model(c)
  sess = gpt2.start_tf_sess()
  gpt2.load_gpt2(sess, model_name=c['model_name'])
  args = dict(
      model_name=c['model_name'], 
      length=c['output_length'],
      return_as_list=True,
  )
  if prefix is not None:
    args['prefix'] = prefix
  return gpt2.generate(sess, **args)[0]