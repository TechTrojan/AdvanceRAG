from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DataRetriever:
    emd_model : str  = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings : HuggingFaceEmbeddings = None 
    
    vec_store : FAISS = None 
    filePath : str = None 
    retriever :VectorStoreRetriever = None 
    
    def __init__(self, index_filePath:str ):
        self.embeddings = HuggingFaceEmbeddings(model_name = self.emd_model)
        self.filePath = index_filePath
        
    def LoadDatabase(self)-> bool : 
        try:
            
            
            self.vec_store = FAISS.load_local(
                        self.filePath,
                        self.embeddings,
                        allow_dangerous_deserialization=True
            )
            
            self.retriever=  self.vec_store.as_retriever(search_type='similarity', search_kwargs={"k":5})
            
            return True         
        except Exception as e:
            print(e)
            return False 

            