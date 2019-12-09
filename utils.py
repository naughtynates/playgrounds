from google.colab import auth
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google.colab import drive
import os

def restart_runtime():
  os.kill(os.getpid(), 9)

def save_to_drive(name, path):
	auth.authenticate_user()
	drive_service = build('drive', 'v3')

	file_metadata = {
		'name': name,
		'mimeType': 'application/octet-stream'
	}

	media = MediaFileUpload(
		path, 
		mimetype='application/octet-stream',
		resumable=True
	)

	created = drive_service.files().create(
		body=file_metadata,
		media_body=media,
		fields='id'
	).execute()
	return created.get('id')

def load_from_drive(name, path):
	drive.mount('/content/drive')
	with open(path, 'rb') as f:
		with open('/content/drive/My Drive/' + name, 'wb') as w:
	  		w.write(f.read())
	drive.flush_and_unmount()

def mount_drive(drive_path='/content/drive/'):
	drive.mount(drive_path)
	return drive_path + 'My Drive/'














