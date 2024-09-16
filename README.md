# 10K Data Analyzer

## Overview

The **10K Data Analyzer** is a Streamlit application that leverages advanced language models and search engines to provide detailed analyses of financial statements and related news. The app uses the Llama3 70B language model, inferenced in Groq, to generate comprehensive insights based on user queries.

## Architecture

1. **Data Retrieval**:
   - **DuckDuckGo Search**: The app performs searches using DuckDuckGo to gather relevant articles and statements based on the user query.

2. **Text Analysis**:
   - **Llama3 70B (Groq)**: The retrieved text is analyzed using the Llama3 70B language model, which is inferenced in Groq. This model generates detailed and contextually rich responses to the user's queries.

3. **Streamlit Interface**:
   - **User Input**: Users enter their queries via an interactive chat interface.
   - **Response Display**: The app displays responses from the language model in a chat format, providing users with detailed analyses of their queries.

## Example Queries

Here are some example queries you can try:

- "Find recent news about Tesla and provide an analysis."
- "Analyze the latest financial statements of Microsoft."
- "What are the key takeaways from Apple's latest quarterly report?"
- "Summarize the news articles about Amazon from this week."

## How It Works

1. **User Query**: Enter your query into the chat input area.
2. **Search and Analysis**:
   - The app searches for relevant articles and statements using DuckDuckGo.
   - It then sends the gathered information to the Llama3 70B model for analysis.
3. **Response**: The app displays the detailed analysis from the Llama3 70B model in the chat interface.

## Deployment

To view the deployed app, click on the link: [View Deployed App](https://10k-chatbot.streamlit.app/)

## Note

- The application is designed to handle various financial queries and provide in-depth analyses based on the latest available data.

