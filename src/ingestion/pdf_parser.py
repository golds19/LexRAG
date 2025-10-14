"""
This section reads the pdf datasets
- Converts the data into chunks
"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from typing import Any, List
from utils.helper_function import clean_text
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# loading all documents from directory.v
def load_from_dir(dir_path:str):
    """Loads all documents from directory"""
    try:
        dir_loader = DirectoryLoader(
        dir_path,
        glob="**/*.pdf", ## patterns to match files
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

        documents = dir_loader.load()

        return documents
    
    except Exception as e:
        print(f"Error occured when loading documents from directory: {e}")


# loading a single document
def load_document(path:str):
    """Load a single document"""
    try:
        pypdf_loader = PyPDFLoader(path)
        pypdf_docs = pypdf_loader.load()

        return pypdf_docs
    
    except Exception as e:
        print("Error occured when loading pdf file: {e}")

# preprocesing and chunking the document
class SmartPDFProcessor:
    """Advanced PDF Processing with error handling"""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size=chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n", " "]
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.semantic_chunker = SemanticChunker(embeddings=self.embeddings)
        self.clean_text = clean_text
        

    def process_pdf(self, docs:List[Document]) -> List[Document]:
        """Process each PDF with smart chunking and metadata enhancement"""
        # Processing each PDF
        processed_chunks = []
        try:
            for idx, doc in enumerate(docs):
                # clean text
                cleaned_text = self.clean_text(doc.page_content)

                # skip nearly empty pages
                if len(cleaned_text.strip()) < 60:
                    continue

                # Create chunks with enhanced metadata
                # chunks = self.text_splitter.create_documents(
                #     texts = [cleaned_text],
                #     metadatas=[{
                #         **doc.metadata,
                #     }]
                # )
                chunks = self.semantic_chunker.create_documents(
                    texts = [cleaned_text],
                    metadatas=[{
                        **doc.metadata,
                    }]
                )

                processed_chunks.extend(chunks)

            return processed_chunks
        
        except Exception as e:
            print(f"Error occured in chunking documents: {e}")

    def create_vector_store_and_retriever(self, docs: List[Document]) -> FAISS:
        """Create a FAISS vector store from processed documents"""
        try:
            vector_store = FAISS.from_documents(docs, self.embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
            return retriever

        except Exception as e:
            print(f"Error occured in creating vector store: {e}")

    def chat_with_pdf(self, retriever:Any, query:str) -> str:
        """Chat with the processed PDF using a retriever and LLM"""
        try:
            template = """Answer the question based on the following context:
            {context}

            Question: {question}
            """
            prompt = PromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)

            # LCEL chain
            rag_chain = (
                RunnableMap({
                    "context": lambda x: retriever.invoke(x["question"]),
                    "question": lambda x: x["question"]
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            response = rag_chain.invoke({"question": query})
            return response
        except Exception as e:
            print(f"Error occured in chatting with PDF: {e}")


if __name__ == "__main__":
    # loading documents from directory
    docs = load_document("Dataset/Less is More_Recursive Reasoning with Tiny Networks.pdf")
    preprocessor = SmartPDFProcessor()
    processed_chunks = preprocessor.process_pdf(docs)
    retriever = preprocessor.create_vector_store_and_retriever(processed_chunks)
    response = preprocessor.chat_with_pdf(retriever, "what are the key insights from the paper?")
    print("Response from chat with PDF:")
    print(response)