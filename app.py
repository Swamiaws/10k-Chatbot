import streamlit as st
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# Initialize the LangChain Groq LLM
llm_groq = ChatGroq(model_name="llama3-70b-8192", api_key=st.secrets["GROQ_API_KEY"])

# Initialize Conversation Buffer Memory
memory = ConversationBufferMemory()

def ten_k_analyzer(query, memory):
    """
    Searches for relevant articles based on the given query and provides a detailed analysis using the Groq LLM.
    """
    # Perform a DuckDuckGo search with the provided query
    text = ""
    results = DDGS().text(query, region='in-en')
    
    for article in results:
        # Add title and body of each article to the text
        title = article.get('title', '')
        body = article.get('body', '')
        if title and body:
            text += f"Title: {title}\nBody: {body}\n\n"

    if not text:
        return "❌ No relevant articles or statements found."

    # Formulate prompt for the Groq LLM based on the query and search results
    prompt = (
        f"Provide a detailed analysis for the following user query: '{query}'. "
        f"Below are the relevant articles and statements gathered from various sources:\n\n{text}"
    )
    
    # Invoke the LLM to generate a detailed analysis
    response = llm_groq.invoke(prompt).content
    
    # Update memory with the current interaction
    memory.add_message("user", query)
    memory.add_message("assistant", response)
    
    return response

# Streamlit app
def main():
    """Main function to run the Streamlit chat interface."""
    st.set_page_config(page_title="10K Data Analyzer", page_icon="📄🔍", layout="wide")
    st.title("10K Data Analyzer")
    st.write("This app searches for relevant Financial Statements across the internet.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # User input area for queries
    if user_query := st.chat_input("Enter your query (e.g., news about a company):"):
        # Add user message to the session
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
    
        # Show the response from the model
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("...")
    
            try:
                response = ten_k_analyzer(user_query, memory)
                placeholder.markdown(response)
                
                # Add assistant message to the session
                st.session_state.messages.append({"role": "assistant", "content": response})
    
            except Exception as e:
                placeholder.markdown(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
