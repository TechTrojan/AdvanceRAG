from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os 
import faiss
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFDataIngester:
    documents= [] 
    dirPath:str= '' 
    chk_size:int = 1000
    chk_overlap:int = 200
    emd_model : str  = 'text-embedding-3-small'
    splitter: RecursiveCharacterTextSplitter  = None 
    chunks : list[Document] = None 
    embeddings : HuggingFaceEmbeddings = None 
    vec_store : FAISS = None 
    
    def __init__(self,dPath,embedding_model):
        self.dirPath = dPath
        self.emd_model= embedding_model
        self.embeddings= HuggingFaceEmbeddings(model_name =self.emd_model)
        
    
    def LoadDocuments(self) :
        for file in os.listdir(self.dirPath):
            docs = self.__LoadDocument__(f"{dirPath}\\{file}")
            if docs is not None :
                self.documents.extend(docs )
        

    def __LoadDocument__(self,filePath):
        docs = None 
        try:
            loader = PyMuPDFLoader(file_path=filePath)

            docs = loader.load()    
        except Exception as e :
            print('exception while loading documents ', str(e))
        return docs 
    
    def init_splitter(self):
        if self.documents is None :
            raise Exception('Documents are not loaded so splitter can''t initialize')
            return 
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chk_size,
            chunk_overlap = self.chk_overlap
            
        )
        
        
    


    def create_chunks(self)  :
        try:
            self.chunks = self.splitter.split_documents(self.documents)
            
        except Exception as e :
            print(e)

    def create_vector_store(self):
        try:
            self.vec_store = FAISS.from_documents(self.chunks, self.embeddings)
        except Exception as e :
            print(e)

    def store_vector_to_local(self) -> str :
        localPath: str = None 
        
        try:
            localPath = f"{self.dirPath}\\faiss_index"    
            self.vec_store.save_local(localPath)
        except Exception as e :
            print(e)





cwd = os.getcwd()





dirPath  = f"{cwd}\\Baseline_Chunking\\data\\"

pdfdata= PDFDataIngester(dirPath, "sentence-transformers/all-MiniLM-L6-v2")
pdfdata.LoadDocuments()
pdfdata.init_splitter()
pdfdata.create_chunks()
pdfdata.create_vector_store()
pdfdata.store_vector_to_local()






        
