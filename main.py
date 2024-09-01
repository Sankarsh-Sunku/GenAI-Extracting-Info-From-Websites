from dotenv import load_dotenv
import streamlit as st
# from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
import pickle
import os
import nltk
import torch

# Download the Punkt tokenizer models
nltk.download('punkt', quiet=True)

load_dotenv()

st.title("News Research Tool")

st.sidebar.title("News Article Units")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    print(url)

process_url_clicked = st.sidebar.button("Process URLS")
file_path = "huggingface-vector.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    #step 1 is to load the text or the data from the urls
    loader = UnstructuredURLLoader(
        urls = urls
    )
    main_placeholder.text("Data Loading...  Started... ")
    data = loader.load()
    print(len(data))
    #split the data into chunks and here we use Recursive Character Split
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
        chunk_size=1000,  # size of each chunk created
        chunk_overlap=200,  # size of  overlap between chunks in order to maintain the context
        # length_function=len : need to know why this is being used as of now we commented this param
        # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
    )
    main_placeholder.text("Splitting Data...  Started... ")
    docs = r_splitter.split_documents(data)
    #in this step we need to do embeddings that means all the chaunks need to be converted into the vectors so that further prediction can be done by the LLM
    embeddings = HuggingFaceEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)
    with open(file_path, 'wb') as f:
        pickle.dump(vector_index, f)
    main_placeholder.text("Converting Text Chunks into Vectors...  Started... ")

query = main_placeholder.text_input("Question :")
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vector_index = pickle.load(f)
            # retrieval-based question-answering (QA) chain using LangChain, a framework designed to build language model applications. The purpose of this chain is to retrieve relevant information from a set of documents (stored in a vector index) and use a language model (LLM) to generate answers to questions based on the retrieved information
            login(token="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")
            repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Name of the model which you are working
            llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, model_kwargs={"max_length": 512,
                                                                                          "token": "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"})
            # as the temperature is increased it will take risks and be more creative to provide the results for our prompt
            chains = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
            result = chains({"question": query}, return_only_outputs=True)
            st.header("Answer : ")
            st.subheader(result["answer"])
