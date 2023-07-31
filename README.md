from langchain.document_loaders import PyPDFLoader
loader=PyPDFLoader('chargeback-guide.pdf')
pages=loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
chunk_size = 1500
chunk_overlap = 150
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
splits=r_splitter.split_documents(pages)
from langchain.embeddings import HuggingFaceEmbeddings
embedding_model=HuggingFaceEmbeddings()
paras=[i.page_content for i in splits]
from langchain.vectorstores import FAISS
db=FAISS.from_texts(sample,embedding_model)
