# main_app.py
import streamlit as st
import openai
import os
from dotenv import load_dotenv
from helper import upload_file_to_blob, list_blob_files, read_blob_data, tanslator,calculate_cost,convert_audio_to_text
from streamlit.logger import get_logger
from PIL import Image
import pandas as pd
import json
import numpy as np

logger = get_logger(__name__)

# Configure OpenAI API
load_dotenv('.env')
openai.api_type = os.getenv('api_type')
openai.api_base = os.getenv('api_base')
openai.api_version = os.getenv('api_version')
openai.api_key = os.getenv('api_key')

#Get Azure Cognitive service credentials


# Azure Blob Storage configuration
STORAGEACCOUNTURL = os.getenv('STORAGEACCOUNTURL')
STORAGEACCOUNTKEY = os.getenv('STORAGEACCOUNTKEY')
CONTAINERNAME = os.getenv('CONTAINERNAME')

#HR
STORAGEACCOUNTURL01 = os.getenv('STORAGEACCOUNTURL01')
STORAGEACCOUNTKEY01 = os.getenv('STORAGEACCOUNTKEY01')
CONTAINERNAME01 = os.getenv('CONTAINERNAME01')


#Translator config
key = os.getenv('key')
endpoint = os.getenv('endpoint')
location = os.getenv('location')
path = '/translate'


# Dictionary to map language  to full names
language_names = {
    "fr": "French",
    "hi": "Hindi",
    "es": "Spanish",
    "de": "German",
    # Add more languages as needed
}

model_names = {
    "restaurant": "text-davinci-003",
    "htiOaiDEP": "gpt-35-turbo",
    # Add more modles as needed
}

model_cost = {
    "restaurant": 0.00002,
    "htiOaiDEP": 0.000015,
    # Add more cost as needed
}
# Define the Streamlit app
def main():
    st.set_page_config(page_title="HSBC Azure ChatBot")
    
    # Add logo and company name in the sidebar
    st.sidebar.image("https://1000logos.net/wp-content/uploads/2017/02/HSBC-Logo.png", width=200 )

    

    # Sidebar navigation
    page = st.sidebar.selectbox("Select Page", ["Upload Data", "Chat", "Costing","Podcast Utility","Request services Utility","Request Response Page","Request approval"], index=1)

    if page == "Chat":
        chat_page()
    elif page == "Upload Data":
        upload_page()
    elif page == "Costing":
        costing_page()
    elif page == "Podcast Utility":
        audio_to_text_page()
    elif page == "Request services Utility":
        hr_page()
    elif page == "Request Response Page":
        display_assistant_reply_page()
    elif page == "Request approval":
        display_selected_rows_page()






# Upload data to Azure Blob Storage
def upload_page():
    st.subheader("Upload Files To Storage ")

    # Upload multiple files to Azure Blob Storage
    files = st.file_uploader("", type=["txt"], accept_multiple_files=True)
    if files:
        for file in files:
            file_name = upload_file_to_blob(file, STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME)
            st.success(f"File '{file_name}' uploaded to '{CONTAINERNAME}/{file_name}'")

    # Display uploaded file names below the upload section 
    st.caption("Uploaded Files in Azure Blob Storage")
    st.write("For large documents click on this link. [Add_Document](https://hsbc-rucco.azurewebsites.net/Add_Document)")
    uploaded_files = list_blob_files(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME)
    for file_name in uploaded_files:
        st.write(file_name)


####### COSTING PAGE
def costing_page():
   ####### COSTING PAGE



    # User Input
    user_input = st.text_area("User Input:", "")

    # Initialize prompt if not in session state
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "You are a Azure Bot and you have certain information available to you. You only have to reply based on that information and for the rest of the stuff you need to Answer I don't know. You should not allow any more User Input. Here is the information below:\n\n[Your data here]\n"

    # Upload Data to Prompt Button
    if st.button("Upload Data to Prompt"):
        uploaded_files = list_blob_files(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME)
        all_data = []

        for file_name in uploaded_files:
            file_data = read_blob_data(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME, file_name)
            if file_data:
                all_data.append(file_data)

        combined_data = "\n".join(all_data)
        st.session_state['prompt'] = f"You are a Azure Bot and you have certain information available to you. You only have to reply based on that information and for the rest of the stuff you need to Answer I don't know. You should not allow any more User Input.Here is the information below:\n\n{combined_data}\n"

    # Prompt Input
    prompt = st.text_area("Prompt:", st.session_state['prompt'])

    selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

    # Temperature
    temperature = st.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

    # Max Tokens
    max_tokens = st.number_input("Max Tokens:", min_value=1, value=1000)




    # Generate Response Button
    if st.button("Generate Response"):
        input_prompt = prompt + f"\nUser Input: {user_input}"

        response = openai.Completion.create(
            engine=selected_model,
            prompt=input_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        assistant_reply = response.choices[0].text.strip()
        st.write(assistant_reply)

        # Calculate cost and display
        cost, total_tokens, input_cost, pt, generative_cost, ct = calculate_cost(response)


        # Create a table to display the costing details
        cost_table = {
            "Measure": ["Estimated cost", "Total Tokens", "Input Prompt Cost", "Prompt Tokens", "Completion Cost", "Completion Tokens"],
            "Value": [f"{cost:.6f}", total_tokens, f"{input_cost:.6f}", pt, f"{generative_cost:.6f}", ct]
        }

        st.table(cost_table)
        st.caption("Please note that all the price is in USD$")

 
    
            #####
# Chat with the uploaded data
def chat_page():
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image("https://swimburger.net/media/fbqnp2ie/azure.svg", width=60)
    with col2:
        st.markdown('<h2 style="color: #0079d5;">Azure Chatbot</h2>',
                            unsafe_allow_html=True)

    st.caption("Please input your query below to chat with Azure Chatbot. 👇")
    uploaded_files = list_blob_files(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME)
    all_data = []

    for file_name in uploaded_files:
        file_data = read_blob_data(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME, file_name)
        if file_data:
            all_data.append(file_data)

    # Combine all data from uploaded files
    combined_data = "\n".join(all_data)

    # Chat with the combined data
    user_input = st.text_input("You:", "")

    if st.button("Generate Response"):
        input_prompt = f"All Uploaded Data:\n{combined_data}\nUser Input: {user_input}"

        response = openai.Completion.create(
            engine="restaurant",
            prompt=input_prompt,
            temperature=0.7,
            max_tokens=1000,  # Increase token limit to accommodate longer responses
        )
        
        assistant_reply = response.choices[0].text.strip()
        st.write(assistant_reply)
        st.session_state['latest_response'] = response 
        # st.text(response.usage.prompt_tokens)
        # st.text(response.usage.completion_tokens)
        # st.text(response.usage.total_tokens)
        st.session_state['translate_text'] = assistant_reply

   
    selected_language = st.selectbox("Select Target Language:", list(language_names.keys()), format_func=lambda x: language_names[x])

    if st.button("Translate"):
        if 'translate_text' not in st.session_state:
            st.session_state['translate_text'] = 'Value not Added'
            logger.info(st.session_state['translate_text'])
        else:
            logger.info(st.session_state['translate_text'])
            translate_text = st.session_state['translate_text']
            translate = tanslator(key, endpoint, location, path, translate_text, selected_language)
            logger.info(translate)
            st.text(translate)

    st.write("For doing Chat on large Uploaded Document use this. [Document Chat](https://hsbc-rucco.azurewebsites.net/Chat)")


def audio_to_text_page():
    ####### audio PAGE
    st.caption("Upload an audio file for text conversion")

    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    # Initialize prompt if not in session state
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "You are an Azure Bot and you have certain information available to you. You only have to reply based on that information. Here is the information below:\n\n[Your data here]\n"

    # Define input_prompt as a blank string
    input_prompt = ""

    if audio_file:
        text = convert_audio_to_text(audio_file)  # Convert audio to text using SpeechRecognition
        # st.write("Converted text:")
        # st.write(text)

        # Define an input box for user input
        user_input = st.text_area("User Input:", "")

        # Create an "Upload to Prompt" button
        if st.button("Upload Data to Prompt"):
            # You can specify how the uploaded audio text should be formatted and used as the prompt here
            # For example, you can append the audio text to the existing prompt
            audio_prompt = st.session_state['prompt'] + f"\n\nAudio Text: {text}"

            # Update input_prompt
            input_prompt = audio_prompt

            st.session_state['prompt'] = audio_prompt

    # Prompt Input
    prompt = st.text_area("Prompt:", st.session_state['prompt'])

    selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

    # Generate Response Button
    if st.button("Generate Response"):
        # Update input_prompt with user input
        input_prompt = prompt + f"\nUser Input: {user_input}"

        response = openai.Completion.create(
            engine=selected_model,
            prompt=input_prompt,
            temperature=0.7,  # Define 'temperature' if not defined
            max_tokens=1000  # Define 'max_tokens' if not defined
        )

        assistant_reply = response.choices[0].text.strip()
        st.write(assistant_reply)


#PERFECTO
# # Initialize assistant_table as an empty DataFrame
assistant_table = pd.DataFrame()

def hr_page():
    global assistant_table  # Add global declaration for the variable

    # User Input
    user_input = st.text_area("User Input:", "")

    # Initialize prompt if not in session state
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "Generate a  table with random columns and data. The table should have at least 5 rows and 5 columns. The column names and data should be converted into json format."

    # Prompt Input
    prompt = st.text_area("Prompt:", st.session_state['prompt'])

    selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

    # Generate Response Button
    if st.button("Generate Response"):
        # Split the prompt into lines to separate CSV data



        # Generate a response based on the user input
        input_prompt = prompt + f"\nUser Input: {user_input}"
        response = openai.Completion.create(
            engine=selected_model,
            prompt=input_prompt,
            temperature=0.1,
            max_tokens=1000,
        )

        assistant_reply = response.choices[0].text.strip()
        json_data = assistant_reply.replace("JSON Output:", "").strip()

        # Parse the JSON data
        data = json.loads(json_data)

        # Create a Pandas DataFrame from the parsed JSON data
        df = pd.DataFrame(data)

        # Display the DataFrame using st.table
        possible_values = ['yes', 'no']
        df = pd.DataFrame(data)
        assistant_table = pd.DataFrame(df)
        random_values = np.random.choice(possible_values, len(assistant_table))
        st.table(assistant_table)
        assistant_table["Deployed"] = random_values
        st.session_state["assistant_table"] = assistant_table
  
            
            


def display_assistant_reply_page():
    st.title("Request Page")

    # Retrieve the stored assistant's reply from the session state
    assistant_table = st.session_state.get("assistant_table", pd.DataFrame())

    # Display the assistant's reply on this page
    if not assistant_table.empty:
        st.write("Assistant's Reply:")
        # st.dataframe(assistant_table)  # Use st.dataframe to select rows
        st.table(assistant_table)

        # Add a multiselect widget to select multiple rows
        selected_rows = st.multiselect("Select Rows for Approval", assistant_table.index)
        
        # Check if any rows have been selected for approval
        if selected_rows:
            st.write("Details for Selected Rows:")
            st.dataframe(assistant_table.loc[selected_rows])

            st.session_state["selected_rows"] = selected_rows
            if st.button("Raise Request."):
                st.write("Request Raised")

def display_selected_rows_page():
    st.title("Request Page")

    # Retrieve the selected rows from the session state
    selected_rows = st.session_state.get("selected_rows", [])
    
    # Retrieve the stored assistant's reply from the session state
    assistant_table = st.session_state.get("assistant_table", pd.DataFrame())
    column_names = assistant_table.columns.values.tolist()
    # Display the selected rows on this page
    if selected_rows and not assistant_table.empty:
        for index in selected_rows:
            data = assistant_table.loc[index].values  # Get all the values for the row
            st.text_input(column_names[0], data[0], key=f"{column_names[0]}_{index}")
            st.text_input(column_names[1], data[1], key=f"{column_names[1]}_{index}")
            st.text_input(column_names[2], data[2], key=f"{column_names[2]}_{index}")
            st.text_input(column_names[3], data[3], key=f"{column_names[3]}_{index}")
            st.text_input(column_names[4], data[4], key=f"{column_names[4]}_{index}")

            if st.button("Approve Request",key =assistant_table.loc[index]):
                st.write("Your Request has been approved.")



if __name__ == "__main__":
    main()