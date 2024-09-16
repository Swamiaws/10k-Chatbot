import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embedding technique
from langchain.vectorstores import FAISS  # Vector Storage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to initialize session state
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_conversational_chain(selected_model, temperature):
    """Creates a conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=selected_model, temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, selected_model, temperature):
    """Handles user questions and generates responses using the FAISS vector store and conversational chain."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the pre-existing FAISS vector store
    file_name = "faiss_index"  # Specify your FAISS index file name here
    new_db = FAISS.load_local(file_name, embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search to find relevant documents
    docs = new_db.similarity_search(user_question)

    # Get the response from the conversational chain
    chain = get_conversational_chain(selected_model, temperature)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Streamlit app
def main():
    """Main function to run the Streamlit chat interface."""
    st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="wide")
    st.title("Document Chatbot")
    st.write("This chatbot answers questions based on the content of a pre-processed PDF file using FAISS and Google Generative AI.")
    
    # Sidebar for selecting LLM and temperature
    with st.sidebar:
        st.subheader("Model Settings")
        
        # Radio button to select LLM
        selected_model = st.radio(
            "Select LLM Model:",
            options=["gemini-pro"],  # Add more models to this list if available
            index=0
        )
        
        # Slider to adjust temperature
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

        # File information
        st.subheader("Data Information")
        file_name = "faiss_index"  # Specify your FAISS index file name here
        st.text(f"File Name: {file_name}")
    
    # Initialize session state
    init_state()

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # User input area for questions
    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        # Add user message to the session
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Show the response from the model
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("...")

            try:
                response = user_input(user_question, selected_model, temperature)
                placeholder.markdown(response)
                
                # Add assistant message to the session
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                placeholder.markdown(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
