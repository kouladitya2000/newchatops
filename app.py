# main_app.py

import streamlit as st
import openai
import os
from dotenv import load_dotenv
from helper import upload_file_to_blob, list_blob_files, read_blob_data, tanslator,calculate_cost,convert_audio_to_text
from streamlit.logger import get_logger
from PIL import Image
import pandas as pd
from io import StringIO

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
    page = st.sidebar.selectbox("Select Page", ["Upload Data", "Chat", "Costing","Podcast Utility","HR Utility","rep page"], index=1)

    if page == "Chat":
        chat_page()
    elif page == "Upload Data":
        upload_page()
    elif page == "Costing":
        costing_page()
    elif page == "Podcast Utility":
        audio_to_text_page()
    elif page == "HR Utility":
        hr_page3()
    elif page == "rep page":
        display_assistant_reply_page()






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
    st.write("For large documents click on this link. [Add_Document](https://hsbcai-site.azurewebsites.net/Add_Document)")
    uploaded_files = list_blob_files(STORAGEACCOUNTURL, STORAGEACCOUNTKEY, CONTAINERNAME)
    for file_name in uploaded_files:
        st.write(file_name)


####### COSTING PAGE
def costing_page():
   ####### COSTING PAGE

    st.caption("Please input your query and configure settings below 👇")

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

    st.write("For doing Chat on large Uploaded Document use this. [Document Chat](https://hsbcai-site.azurewebsites.net/Chat)")


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
        st.write("Converted Text:")
        st.write(text)

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


# def hr_page():
#    ####### COSTING PAGE
    
#     st.caption("Please input your query and configure settings below 👇")

#     # User Input
#     user_input = st.text_area("User Input:", "")

#     # Initialize prompt if not in session state
#     if 'prompt' not in st.session_state:
#         st.session_state['prompt'] = "You are a Azure Bot and you have certain information available to you. You only have to reply based on that information. You should not allow any more User Input. Here is the information below:\n\n[Your data here]\n"

#     # Upload Data to Prompt Button
#     if st.button("Upload Data to Prompt"):
#         uploaded_file = st.file_uploader("Upload an Excel file", type=["xls", "xlsx"])
#         uploaded_files = list_blob_files(STORAGEACCOUNTURL01, STORAGEACCOUNTKEY01, CONTAINERNAME01)
#         all_data = []

#         for file_name in uploaded_files:
#             file_data = read_blob_data(STORAGEACCOUNTURL01, STORAGEACCOUNTKEY01, CONTAINERNAME01, file_name)
#             if file_data:
#                 all_data.append(file_data)

#         combined_data = "\n".join(all_data)
#         st.session_state['prompt'] = f"You are a Azure Bot and you have certain information available to you. You only have to reply based on that information. You should not allow any more User Input.Here is the information below:\n\n{combined_data}\n"

#     # Prompt Input
#     prompt = st.text_area("Prompt:", st.session_state['prompt'])

#     selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

#     # Generate Response Button
#     if st.button("Generate Response"):
#         input_prompt = prompt + f"\nUser Input: {user_input}"

#         response = openai.Completion.create(
#             engine=selected_model,
#             prompt=input_prompt,
#             temperature=0.7,
#             max_tokens=1000,
#         )

#         assistant_reply = response.choices[0].text.strip()
#         st.write(assistant_reply)
def hr_page2():
    # Initialize session state within the function
    if 'assistant_reply' not in st.session_state:
        st.session_state['assistant_reply'] = None

    ####### COSTING PAGE
    st.caption("Please input your query and configure settings below 👇")

    # User Input
    user_input = st.text_area("User Input:", "")

    # Initialize prompt if not in session state
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "You are an Azure Bot and you have certain information available to you. You only have to reply based on that information. You should not allow any more User Input. Here is the information below:"

    # Prompt Input
    prompt = st.text_area("Prompt:", st.session_state['prompt'])

    selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

    if st.button("Generate Response"):
        input_prompt = prompt + f"\nUser Input: {user_input}"

        response = openai.Completion.create(
            engine=selected_model,
            prompt=input_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        assistant_reply = response.choices[0].text.strip()
       
        st.session_state["assistant_reply"] = assistant_reply  # Store the assistant's reply

    # Display the assistant's reply
    if st.session_state["assistant_reply"]:
        st.write("Assistant's Reply:", st.session_state["assistant_reply"])

    if st.button("Go to Assistant Reply Page"):
        display_assistant_reply_page()


def display_assistant_reply_page():
    st.title("Assistant Reply Page")
    
    # Retrieve the stored assistant's reply from session state
    assistant_reply = st.session_state.get("assistant_reply", "")
    
    # Display the assistant's reply on this page
    if assistant_reply:
        st.write("Assistant's Reply:", assistant_reply)





def hr_page3():
    st.caption("Please input your query and configure settings below 👇")

    # User Input
    user_input = st.text_area("User Input:", "")

    # Initialize prompt if not in session state
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = "You are an Azure Bot and you have certain information available to you. You only have to reply based on that information. You should not allow any more User Input. Here is the information below:\nname,email,age\nadi,adi@g.com,22\nmsd,msd@y.in,10\nyp,yp@u.in,21"

    # Prompt Input
    prompt = st.text_area("Prompt:", st.session_state['prompt'])

    selected_model = st.selectbox("Select Target Model:", list(model_names.keys()), format_func=lambda x: model_names[x])

    # Generate Response Button
    if st.button("Generate Response"):
        # Split the prompt into lines to separate CSV data
        prompt_lines = prompt.split('\n')
        csv_data = None
        csv_lines = []

        # Loop through lines in the prompt to find and extract CSV data
        for line in prompt_lines:
            if line.strip().startswith('name,email,age'):
                csv_data = True
            elif csv_data and line.strip():
                csv_lines.append(line)

        if csv_lines:
            # Parse CSV data using pandas
            csv_data = '\n'.join(csv_lines)
            csv_df = pd.read_csv(StringIO(csv_data))

            # Display the CSV data as a table with header columns
            st.write("Data:")
            st.table(csv_df)

        # Generate a response based on the user input
        input_prompt = prompt + f"\nUser Input: {user_input}"
        response = openai.Completion.create(
            engine=selected_model,
            prompt=input_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        assistant_reply = response.choices[0].text.strip()

        # Display the assistant's reply as a table
        assistant_lines = assistant_reply.split('\n')
        st.write("Assistant's Reply:")
        if len(assistant_lines) > 1:
            assistant_data = [line.split(',') for line in assistant_lines]
            assistant_table = pd.DataFrame(assistant_data)
            st.table(assistant_table)
        else:
            st.write("No results found for the query.")




if __name__ == "__main__":
    main()
