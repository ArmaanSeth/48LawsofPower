import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.cassandra import Cassandra
import cassio

path="./48lawsofpower.pdf"
# docs=os.listdir(path)

def get_text_chunks(pdf):
    text=""
    pdf=PdfReader(pdf)
    for page in pdf.pages:
        text+=page.extract_text()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

chunks=get_text_chunks(path)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})

load_dotenv()
cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))

astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="laws_of_power",
    session=None,
    keyspace=None
)

astra_vector_store.add_texts(chunks)