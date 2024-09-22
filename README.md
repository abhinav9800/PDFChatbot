Conversational AI with PDF Upload and History-Aware Chatbot

This project implements a conversational AI chatbot powered by the Ollama Llama 3.1:8b model, which is used both as the large language model (LLM) and for generating embeddings. The chatbot is designed to handle PDF uploads and provide intelligent, context-aware responses based on document content and prior conversation history.

The system is built using LangChain components and integrates a Chroma vector store to manage embeddings. It also includes a Streamlit-based user interface that runs on Google Colab, accessible via Localtunnel.

Key Features

Ollama Llama 3.1:8b as LLM & Embeddings: The Llama 3.1:8b model is used both for generating responses and for producing document embeddings, streamlining the system architecture.

PDF Upload Support: Users can upload PDF documents to the chatbot. The content is automatically processed, split into manageable chunks, and indexed for retrieval.

Chroma Vector Store: The vector store is powered by Chroma, where all embeddings are stored locally in a specified directory, ensuring fast retrieval and seamless interaction.

History-Aware Chatbot: The chatbot is designed to be contextually aware, meaning it takes into account previous interactions in the conversation to better understand and answer user queries.

Streamlit UI on Colab via Localtunnel: The app uses Streamlit to provide a simple, interactive user interface. When deployed on Google Colab, Localtunnel is used to make the application publicly accessible.


Launch the Streamlit App: Run the Streamlit app using the following command:

!streamlit run app.py & npx localtunnel --port 8501

Get localtunnel password using

!curl https://loca.lt/mytunnelpassword


