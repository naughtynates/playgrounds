from google.colab import auth
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google.colab import drive
import os
from IPython.display import HTML, display

def restart_runtime():
  os.kill(os.getpid(), 9)

def save_to_drive(local_path, drive_path):
	auth.authenticate_user()
	drive_service = build('drive', 'v3')

	file_metadata = {
		'name': drive_path,
		'mimeType': 'application/octet-stream'
	}

	media = MediaFileUpload(
		local_path, 
		mimetype='application/octet-stream',
		resumable=True
	)

	created = drive_service.files().create(
		body=file_metadata,
		media_body=media,
		fields='id'
	).execute()
	return created.get('id')

def load_from_drive(drive_path, local_path):
	drive.mount('/content/drive')
	with open('/content/drive/My Drive/' + drive_path, 'rb') as f:
		with open(local_path, 'wb') as w:
	  		w.write(f.read())
	drive.flush_and_unmount()

def mount_drive(drive_path='/content/drive/'):
	drive.mount(drive_path)
	return drive_path + 'My Drive/'

def wrap_text():
	def set_css():
		display(HTML('''
			<style>
				pre {
					white-space: pre-wrap;
				}
			</style>
		'''))
	get_ipython().events.register('pre_run_cell', set_css)










