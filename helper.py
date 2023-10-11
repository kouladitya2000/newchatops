import os
from azure.storage.blob import BlobServiceClient
import requests
import uuid
import speech_recognition as sr



def upload_file_to_blob(file, STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME):
    """
    Uploads a file to Azure Blob Storage.
    
    Parameters:
        file: Uploaded file object from Streamlit.
        STORAGEACCOUNTURL: Azure Storage Account URL.
        STORAGEACCOUNTKEY: Azure Storage Account Key.
        CONTAINERNAME: Azure Blob Storage container name.

    Returns:
        str: The name of the uploaded file.
    """
    if file:
        file_name = file.name

        blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

        blob_client_instance = blob_service_client_instance.get_blob_client(container=CONTAINERNAME, blob=file_name)

        with file as data:
            blob_client_instance.upload_blob(data, overwrite=True)

        return file_name

def read_blob_data(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME, file_name):
    """
    Reads and returns data from Azure Blob Storage.

    Parameters:
        STORAGEACCOUNTURL: Azure Storage Account URL.
        STORAGEACCOUNTKEY: Azure Storage Account Key.
        CONTAINERNAME: Azure Blob Storage container name.
        file_name: Name of the blob to read.

    Returns:
        str: The contents of the blob as a string.
    """
    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    blob_client_instance = blob_service_client_instance.get_blob_client(container=CONTAINERNAME, blob=file_name)
    blob_data = blob_client_instance.download_blob()
    data = blob_data.readall().decode('utf-8')
    return data


def list_blob_files(storage_account_url, storage_account_key, container_name):

    """
Lists and returns the names of blob files within an Azure Blob Storage container.

Parameters:
    storage_account_url (str): The URL of the Azure Blob Storage account.
    storage_account_key (str): The access key for the Azure Blob Storage account.
    container_name (str): The name of the Azure Blob Storage container to list files from.

Returns:
    list of str: A list of file names (blob names) within the specified container.
    An empty list is returned if there are no files or an error occurs.


"""
    try:
        blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=storage_account_key)
        container_client = blob_service_client.get_container_client(container_name)

        blob_files = []

        for blob in container_client.list_blobs():
            blob_files.append(blob.name)

        return blob_files
    except Exception as e:
        print(f"Error listing blob files: {str(e)}")
        return []
    
def tanslator(key, endpoint, location, path, text_content, target_language):
    try:
        constructed_url = endpoint + path

        params = {
            'api-version': '3.0',
            'from': 'en',
            'to': [target_language]  # Use the selected target language
        }

        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        body = [{
            'text': text_content
        }]

        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()

        # Extract the translated text from the response
        return response[0]['translations'][0]['text']

    except Exception as e:
        return str(e)
    
def calculate_cost(response, cost_per_token=0.00002):
    """
    Calculate the cost of the OpenAI API response based on the number of tokens.

    Parameters:
        response: The OpenAI API response object.
        cost_per_token (float): Based on the OpenAI website, the text-davinci-003 model is $0.02 per 1000 tokens

    Returns:
        float: The calculated cost and other details adding up to cost
    """
    try: 
        usage = response.usage
        pt=usage.prompt_tokens #can be removed
        ct=usage.completion_tokens #can be removed
        input_cost=pt * cost_per_token #can be removed
        generative_cost=ct * cost_per_token #can be removed
        total_tokens = usage.total_tokens 
        cost = total_tokens * cost_per_token

        return cost,total_tokens,input_cost,pt,generative_cost,ct
    except AttributeError:
        return 0.0
    


#AUDIO TO TEXT
def convert_audio_to_text(audio_file_path):
    r = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"


