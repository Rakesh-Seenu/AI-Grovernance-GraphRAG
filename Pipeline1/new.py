import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import streamlit as st
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import SentenceTransformer
import uuid

# Initialize ChromaDB client
client = chromadb.Client()

# Web scraping logic to extract articles from URLs
def scrape_articles():
    all_countries = ['India']
    urls = [f"https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-{each}" for each in all_countries]

    # Initialize an empty list to store all the scraped data
    scraped_data = []

    for country, url in zip(all_countries, urls):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find_all('div', class_='field field--name-body field--type-text-with-summary field--label-hidden field--item')
            current_heading = None
            country_data = {'Country': country, 'Sections': []}
            for section in content:
                paragraphs = section.find_all(['p', 'h2', 'h3', 'li'])
                for paragraph in paragraphs:
                    text = paragraph.get_text(strip=True)
                    if paragraph.name in ['h2', 'h3']:
                        current_heading = text
                        country_data['Sections'].append({'Heading': current_heading, 'Content': []})
                    else:
                        if current_heading:
                            country_data['Sections'][-1]['Content'].append(text)
                        else:
                            if not country_data['Sections']:
                                country_data['Sections'].append({'Heading': 'Introduction', 'Content': []})
                            country_data['Sections'][0]['Content'].append(text)
            scraped_data.append(country_data)
        else:
            print(f"Failed to retrieve the webpage for {country}. Status code: {response.status_code}")
    return scraped_data

# Add scraped data to ChromaDB
def add_to_chromadb(scraped_data):
    # Try to get the existing collection or create a new one
    collection_name = "articles1"
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Failed to get collection '{collection_name}': {e}")
        collection = client.create_collection(name=collection_name)

    # Load Sentence Transformer for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Add articles to ChromaDB
    for i, article in enumerate(scraped_data):
        full_text = "\n".join([
            f"{section['Heading']}\n" + "\n.join(section['Content'])"
            for section in article['Sections']
        ])
        
        embedding = model.encode(full_text).tolist()

        collection.add(
            documents=[full_text],
            embeddings=[embedding],
            metadatas=[{'title': article['Country']}],
            ids=[str(i)]
        )

    print(f"Stored {len(scraped_data)} articles in ChromaDB.")
    return collection_name

# Process scraped articles into ChromaDB
def process_articles(scraped_data):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    
    # Process and split the document text
    chunks = []
    for article in scraped_data:
        document_text = "\n".join([
            f"{section['Heading']}\n" + "\n".join(section['Content'])
            for section in article['Sections']
        ])
        document = Document(page_content=document_text)
        chunks += text_splitter.split_documents([document])
    
    # Generate unique IDs for each chunk
    chunk_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Initialize the vector database with the split chunks and the generated IDs
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        ids=chunk_ids,
        collection_name="local-rag"
    )

    return vector_db

# Chain to retrieve and generate responses
def initialize_chain(vector_db):
    local_model = "llama3"
    llm = ChatOllama(model=local_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your sole task is to retrieve the most relevant information 
        strictly from a vector database containing stored articles and content. Your response must be based **only** on 
        the information retrieved from the provided documents. You are NOT allowed to use any external knowledge, personal 
        opinions, or make assumptions. Use the retrieved information strictly to answer the question in the most accurate 
        way possible. 

        If the relevant information is not present in the retrieved content, say explicitly: "The answer is not found in 
        the documents provided."

        Here are ten alternative ways to phrase the original question to improve retrieval performance and ensure the highest 
        likelihood of retrieving the correct answer.

        Original question: {question}
        """
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# Main Streamlit application
def main():
    st.title("AI Governance Tracker Chatbot")
    
    # Scrape and process articles
    st.write("Scraping articles...")
    scraped_data = scrape_articles()
    st.write("Scraping complete. Adding to ChromaDB...")
    
    collection_name = add_to_chromadb(scraped_data)
    
    # Process articles into chunks for ChromaDB
    vector_db = process_articles(scraped_data)
    
    # Initialize the LLM retrieval chain
    chain = initialize_chain(vector_db)
    
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if question and chain:
            response = chain.invoke(question)
            st.write("Response:", response)
        else:
            st.write("Please enter a question.")
    
    if st.button("Cleanup"):
        client.delete_collection(name=collection_name)
        st.write("ChromaDB cleaned up.")

if __name__ == "__main__":
    main()
