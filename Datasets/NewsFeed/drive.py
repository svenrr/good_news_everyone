# pip install PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

gauth = GoogleAuth()
#g_login.LocalWebserverAuth()

# Try to load saved client credentials
gauth.LoadCredentialsFile("credentials.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
#gauth.SaveCredentialsFile("credentials.txt")

drive = GoogleDrive(gauth)


# Get path ID of following path
def get_id(path):
    """
    input path as string, returns fileID
    """
    fileID = 'root'
    for f in path.split('/'):
        fileList = drive.ListFile({'q': "'{}' in parents and trashed=false".format(fileID)}).GetList()
        fileID = None
        for file in fileList:
            #print('Title: %s, ID: %s' % (file['title'], file['id']))
            # Get the folder ID that you want
            if(file['title'] == f):
                fileID = file['id']

    return fileID


# Load content of file from drive
def get_content(path):
    """
    Input path as string, returns content as string
    """
    fileID = get_id(path)
    gfile = drive.CreateFile({'id': fileID})
    return gfile.GetContentString()


# Save Content to drive
def save_to_file(path, content):
    """
    Input path as string, including file name, and input content as string
    """
    fileID = get_id(path)
    # Check if file exits
    if fileID != None:
        gfile = drive.CreateFile({'id': fileID})
        gfile.SetContentString(content)
    else:
        path, title = path.rsplit('/', 1)
        fileID = get_id(path)

        gfile = drive.CreateFile({'title': title,
                                       'parents': [{'id': fileID}]})
        gfile.SetContentString(content)
    gfile.Upload()


# Export a file as json to drive
def export_json(path, file):
    """
    Input path as string, including file name, and input path of file to upload as string
    """
    fileID = get_id(path)
    # Check if file exits
    if fileID != None:
        gfile = drive.CreateFile({'id': fileID})
        gfile.SetContentFile(file)
    else:
        path, title = path.rsplit('/', 1)
        fileID = get_id(path)

        gfile = drive.CreateFile({'title': title,
                                       'parents': [{'id': fileID}]})
        gfile.SetContentFile(file)
    gfile.Upload()