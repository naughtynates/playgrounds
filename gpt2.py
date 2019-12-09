import gpt_2_simple as gpt2
import os

def restart_runtime():
  os.kill(os.getpid(), 9)

def check_model():
  if not os.path.isdir(os.path.join("models", model_name)):
	  gpt2.download_gpt2(model_name=model_name)

def generate(prefix=None):
  check_model()
  sess = gpt2.start_tf_sess()
  gpt2.load_gpt2(sess, model_name=model_name)
  args = dict(
      model_name=model_name, 
      length=output_length,
      return_as_list=True,
  )
  if prefix is not None:
    args['prefix'] = prefix
  return gpt2.generate(sess, **args)[0]