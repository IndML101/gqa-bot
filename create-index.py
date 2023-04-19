import os
from llama_index import SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

class CreateIndex:
    def __init__(self, doc_path='docs') -> None:
        self._loader = SimpleDirectoryReader(doc_path)
        self.documents = [doc.to_langchain_format() for doc in self._loader.load_data()]
        self.embeddings = HuggingFaceEmbeddings(model_name="google/flan-t5-small")

    def split_text(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(self.documents)

    def create_vector_db(self):
        docs = self.split_text()
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local("metamorphosis_index")
        return

if __name__ == '__main__':
    CreateIndex().create_vector_db()