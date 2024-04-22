import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-2-7b-chat-hf', max_length=256, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)

class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        db = FAISS.from_documents(data, embeddings)

        DB_FAISS_PATH = 'vectorstore/db_faiss'

        db.save_local(DB_FAISS_PATH)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):  
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask LLAMA2 about " + self.uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to PDF!", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
