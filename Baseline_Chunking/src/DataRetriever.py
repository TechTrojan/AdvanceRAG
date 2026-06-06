from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from random import sample

class DataRetriever:
    emd_model : str  = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings : HuggingFaceEmbeddings = None 
    
    vec_store : FAISS = None 
    filePath : str = None 
    retriever :VectorStoreRetriever = None 
    docs:list[Document]= None 
    all_docs:list[Document] = None 
    
    def __init__(self, index_filePath:str ):
        self.embeddings = HuggingFaceEmbeddings(model_name = self.emd_model)
        self.filePath = index_filePath
    
    def retrieve_context(self,question)->str:
        self.docs = self.retriever.invoke(question)
        context_list = [ d.page_content for d in self.docs ]
        context = ". ".join(context_list)
        return context
    
    def retrieve_context_list(self,question)->list[str]:
        self.docs = self.retriever.invoke(question)
        context_list = [ d.page_content for d in self.docs ]
         
        return context_list
        
    def     LoadDatabase(self)-> bool : 
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
    
    def RetriveAllDocs(self)-> list[Document] :        
        return  list(self.vec_store.docstore._dict.values())


    def RetriveRandomDocs(self, count: int) -> list[Document]:
        docs = list(self.vec_store.docstore._dict.values())

        if count >= len(docs):
            return docs

        return sample(docs, count)            