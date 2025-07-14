import os
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF for better PDF text extraction
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers.pipelines import pipeline
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key and SerpAPI key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Creating custom template to guide LLM model
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extracting text from PDF using PyMuPDF (fitz)
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as pdf_reader:
            for page_num in range(len(pdf_reader)):
                page_text = pdf_reader[page_num].get_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"No text found on page {page_num + 1}. It may be an image or empty.")
    if not text:
        st.warning("No text extracted from PDFs.")
    return text

# Extracting text from URL
def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        if not text.strip():
            st.warning("No text found on the web page.")
        return text
    except Exception as e:
        st.error(f"Error fetching or parsing the URL: {str(e)}")
        return ""

# Converting text to chunks
def get_chunks(raw_text):
    if not raw_text.strip():
        st.warning("No text available to chunk.")
        return []

    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    if not chunks:
        st.warning("No chunks created. The text may be too short or incorrectly formatted.")
    return chunks

# Using OpenAI embeddings and FAISS to get vectorstore
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Generating conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory)
    return conversation_chain

# Generating response from user queries and displaying them accordingly
def handle_question(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"<h4 style='font-size: 24px;'>User: {msg.content}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='font-size: 24px;'>Bot: {msg.content}</h4>", unsafe_allow_html=True)

# Summarizing text using Hugging Face Transformers
def summarize_text_transformers(text):
    try:
        summarizer = pipeline("summarization")
        if not text.strip():
            return "No text available for summarization."

        summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing the text: {str(e)}"

# Searching the web using SerpAPI
def search_web_serpapi(query):
    try:
        # Construct the search URL for SerpAPI
        search_url = f"https://serpapi.com/search?engine=google&q={query}&api_key={SERPAPI_KEY}&num=5"  # Get top 5 results
        
        # Perform the search
        response = requests.get(search_url)
        response.raise_for_status()  # Check for request errors

        # Parse the search results
        data = response.json()
        results = data.get('organic_results', [])

        if not results:
            st.warning("No search results found.")
            return "No results found for your query."

        # Extract and combine the snippets from multiple search results
        combined_result = "\n\n".join(result.get('snippet', 'No snippet available') for result in results)
        return combined_result
        
    except Exception as e:
        st.error(f"Error searching the web: {str(e)}")
        return "An error occurred while searching the web."

# Frontend modifications with background images and styling
def set_background_image(image_url):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def set_custom_styles():
    st.markdown(""" 
    <style>
    .header {
        text-align: center;
        color: black;
        font-weight: bold;
        font-size: 36px;  /* Increased font size */
        background-color: rgba(255, 255, 255, 0.7);
        padding: 10px;
        border-radius: 10px;
    }
    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
        font-size: 24px;  /* Increased font size */
    }}
    .stFileUploader label {{
        font-size: 25px;  /* Increased font size for labels */
        font-weight: bold;
    }}
    .stTextInput label {{
        font-size: 25px;  /* Increased font size for labels */
        font-weight: bold;
    }}
    .custom-title {
        text-align: center;
        font-weight: bold;
        font-size: 48px;  /* Increased font size */
        color: black;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function
def main():
    # Set page config with default title and layout
    st.set_page_config(page_title="Chat with PDF Documents", layout="wide")

    # Define titles and images for each source type
    titles = {
        "PDF": "PDFs aren’t boring anymore, let’s chat!📚",
        "URL": "Web pages got the info, I’ve got the chat!🌍",
        "Internet Search": "Internet search, but cooler!🔎"
    }

    bg_images = {
        "PDF": 'https://img.freepik.com/free-photo/background-torn-colorful-paper_23-2147734444.jpg?size=626&ext=jpg&ga=GA1.1.52452323.1726288445',
        "URL": 'https://img.freepik.com/free-photo/layered-bright-papers_23-2147734484.jpg?size=626&ext=jpg&ga=GA1.1.52452323.1726288445',
        "Internet Search": 'https://img.freepik.com/free-photo/colorful-ragged-sheets-paper_23-2147734479.jpg?size=626&ext=jpg&ga=GA1.1.52452323.1726288445'
    }
    
    set_custom_styles()
    
    # Set the custom title style for the center-aligned bold text
    st.markdown("""
    <div class="custom-title">FILE, WEB, OR SEARCH – YOUR PICK, WE CHAT!</div>
    """, unsafe_allow_html=True)
    
    # Select source type
    st.markdown('<h2 style="font-size: 30px;">Pick a Source for Your Query </h2>', unsafe_allow_html=True)
    source_type = st.selectbox("", ["PDF", "URL", "Internet Search"], key='source')

    # Update page content based on source type
    st.markdown(f'<h1 style="text-align: center;">{titles[source_type]}</h1>', unsafe_allow_html=True)
    set_background_image(bg_images[source_type])
    
    if source_type == "PDF":
        st.markdown('<label style="font-size: 25px;">SEND ME YOUR PDF SPARKLE!</label>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_chunks(raw_text)
            if chunks:
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    elif source_type == "URL":
        st.markdown('<label style="font-size: 25px;">GOT A LINK? GIVE IT TO ME!</label>', unsafe_allow_html=True)
        url = st.text_input("URL", label_visibility="collapsed")
        if url:
            raw_text = get_url_text(url)
            chunks = get_chunks(raw_text)
            if chunks:
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    elif source_type == "Internet Search":
        st.markdown('<label style="font-size: 25px;">ANY QUERY IN MIND?</label>', unsafe_allow_html=True)
        query = st.text_input("Search Query", label_visibility="collapsed")
        if query:
            search_result = search_web_serpapi(query)
            chunks = get_chunks(search_result)
            if chunks:
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    if st.session_state.get('conversation', None):
        st.markdown('<h2 style="font-size: 30px;">Ask a question related to your uploaded content!</h2>', unsafe_allow_html=True)
        user_question = st.text_input("Question", label_visibility="collapsed")
        if user_question:
            handle_question(user_question)

if __name__ == '__main__':
    main()