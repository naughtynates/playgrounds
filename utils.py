import os

def restart_runtime():
  os.kill(os.getpid(), 9)