import streamlit as st
import speech_recognition as sr
import nltk  # if needed for further processing
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download

# Set up the Streamlit page with a title and header.
st.set_page_config(page_title="Speech-Enabled Chatbot")
st.header("Speech-Enabled Chatbot")

# Text area for entering the system prompt that defines the chatbot's personality.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt"
)

# Initialize the speech recognizer from the SpeechRecognition library.
r = sr.Recognizer()

def transcribe_speech(api_choice, language):
    """
    Transcribe speech input using the selected API and language.
    
    Parameters:
        api_choice (str): The chosen speech recognition API ("google" or "sphinx").
        language (str): The language code for transcription (e.g., "en-US").
        
    Returns:
        str: The transcribed text or an error message.
    """
    with sr.Microphone() as source:
        st.info("Speak now...")  # Inform the user to start speaking.
        r.adjust_for_ambient_noise(source)  # Adjust microphone settings for ambient noise.
        try:
            # Listen for speech input with a timeout.
            audio_text = r.listen(source, timeout=5)
            st.info("Transcribing...")  # Notify user that transcription is in progress.
            # Choose the appropriate API for transcription.
            if api_choice == "google":
                text = r.recognize_google(audio_text, language=language)
            elif api_choice == "sphinx":
                text = r.recognize_sphinx(audio_text, language=language)
            else:
                text = "Selected API not supported yet."
            return text
        except sr.WaitTimeoutError:
            return "No speech detected. Please try again."
        except sr.UnknownValueError:
            return "Sorry, could not understand the audio."
        except sr.RequestError as e:
            return f"API request error: {str(e)}"

@st.cache_resource
def create_chain(system_prompt):
    """
    Load the language model from Hugging Face Hub and create a chatbot chain using LlamaCpp.
    
    Parameters:
        system_prompt (str): The system prompt defining the chatbot's personality.
        
    Returns:
        function: A chain function that takes inputs and returns the chatbot's response.
    """
    # Define the repository and model file name.
    repo_id = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file_name = "llama-2-7b-chat.Q2_K.gguf"
    # Download the model file from the Hugging Face Hub.
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name, repo_type="model")

    # Initialize the LlamaCpp language model with desired parameters.
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0,        # Deterministic output.
        max_tokens=256,       # Maximum number of tokens to generate.
        top_p=1,              # Top-p sampling parameter.
        stop=["</s>"],        # Stop generation on end-of-sequence token.
        verbose=False,
        streaming=True,       # Enable streaming responses.
    )

    # Create a prompt template to format the conversation.
    template = (
        "System: {system_prompt}\n\n"
        "User: {question}\n\n"
        "Assistant:"
    )
    prompt = PromptTemplate(template=template, input_variables=["system_prompt", "question"])

    # Define the chain function that formats the prompt and calls the LLM.
    def chain_fn(inputs):
        formatted_prompt = prompt.format(**inputs)
        return llm(formatted_prompt)

    return chain_fn

# Initialize the chatbot chain using the provided system prompt.
llm_chain = create_chain(system_prompt)

# Initialize session state to store the conversation history.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

# Display the conversation history in the chat interface.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Provide an option for the user to choose between text and speech input.
input_method = st.radio("Choose input method", ["Text", "Speech"])

user_input = None  # Variable to hold the user input.

if input_method == "Text":
    # If text input is chosen, display a chat input box.
    user_input = st.chat_input("Type your message here", key="user_text_input")
elif input_method == "Speech":
    # Define available speech recognition APIs.
    RECOGNITION_APIS = {
        "Google Speech Recognition": "google",
        "CMU Sphinx (Offline)": "sphinx"
    }
    # Define available languages for transcription.
    LANGUAGES = {
        "English (US)": "en-US",
        "English (UK)": "en-GB",
        "French": "fr-FR",
        "Spanish": "es-ES",
        "German": "de-DE"
    }
    # Let the user select the desired speech recognition API.
    api_choice = st.selectbox("Select Speech Recognition API", list(RECOGNITION_APIS.keys()))
    api_choice_key = RECOGNITION_APIS[api_choice]
    # Let the user select the language.
    language_name = st.selectbox("Select Language", list(LANGUAGES.keys()))
    language = LANGUAGES[language_name]
    # Button to start recording speech.
    if st.button("Start Recording"):
        transcribed_text = transcribe_speech(api_choice_key, language)
        user_input = transcribed_text  # Set the transcribed speech as user input.
        st.write("Transcription:", transcribed_text)

# If user input (either text or speech) is available, process it.
if user_input:
    # Append the user's message to the conversation history.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a response from the chatbot using the LLM chain.
    response = llm_chain({"system_prompt": system_prompt, "question": user_input})
    # Append the chatbot's response to the conversation history.
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
