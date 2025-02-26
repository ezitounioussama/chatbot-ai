import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download

# Set the webpage title and header.
st.set_page_config(page_title="Llama AI Chatbot!")
st.header("Your own aiChat with Llama!")

# System prompt for the LLM's personality.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt"
)

@st.cache_resource
def create_chain(system_prompt):
    repo_id = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file_name = "llama-2-7b-chat.Q2_K.gguf"
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name, repo_type="model")

    # Initialize the LlamaCpp LLM.
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=256,  # Ensure that prompt+response tokens fit within the model's context window.
        top_p=1,
        stop=["</s>"],  # Adjust the stop tokens as needed for your model.
        verbose=False,
        streaming=True,
    )

    # Create a prompt template.
    template = (
        "System: {system_prompt}\n\n"
        "User: {question}\n\n"
        "Assistant:"
    )
    prompt = PromptTemplate(template=template, input_variables=["system_prompt", "question"])

    # Define a custom chain function that formats the prompt and calls the LLM.
    def chain_fn(inputs):
        formatted_prompt = prompt.format(**inputs)
        return llm(formatted_prompt)

    return chain_fn

llm_chain = create_chain(system_prompt)

# Initialize session state to store conversation history.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# Render the conversation from session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Take user input and pass it to the custom chain.
if user_prompt := st.chat_input("Your message here", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate the response using the custom chain.
    response = llm_chain({"system_prompt": system_prompt, "question": user_prompt})

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
