import os

def restart_runtime():
  os.kill(os.getpid(), 9)

def save_to_drive(name, path):
	from google.colab import auth
	from googleapiclient.http import MediaFileUpload
	from googleapiclient.discovery import build

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