import gpt_2_simple as gpt2s
import os

def restart_runtime():
  os.kill(os.getpid(), 9)

def check_model(c):
  if not os.path.isdir(os.path.join("models", c['model_name'])):
	  gpt2s.download_gpt2(model_name=c['model_name'])

def generate(c):
  check_model(c)
  sess = gpt2s.start_tf_sess()
  gpt2s.load_gpt2(sess, model_name=c['model_name'])
  args = dict(
      model_name=c['model_name'], 
      length=c['output_length'],
      return_as_list=True,
  )
  if c['prefix'] is not None:
    args['prefix'] = prefix
  return gpt2s.generate(sess, **args)[0]