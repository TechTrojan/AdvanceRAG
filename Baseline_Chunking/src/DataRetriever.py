from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from  langchain_classic.retrievers  import  ContextualCompressionRetriever

class DataRetriever:
    emd_model : str  = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings : HuggingFaceEmbeddings = None 
    
    bm25 : BM25Retriever = None
    ensembel_retriever : EnsembleRetriever = None 
    
    vec_store : FAISS = None 
    filePath : str = None 
    retriever :VectorStoreRetriever = None 
    docs:list[Document]= None 
    allDocuments:list[Document] = None 
    crossencoder : HuggingFaceCrossEncoder = None 
    
    reranker : CrossEncoderReranker = None 
    
    reranker_retriever: ContextualCompressionRetriever = None 
    
    
    def __init__(self, index_filePath:str ):
        self.embeddings = HuggingFaceEmbeddings(model_name = self.emd_model)
        self.filePath = index_filePath
        
        self.crossencoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = CrossEncoderReranker(model=self.crossencoder, top_n=10)
        
        
            
        
    
    def retrieve_context(self,question)->str:
        #self.docs = self.ensembel_retriever.invoke(question)
        self.docs = self.reranker_retriever.invoke(question)
        
        context_list = [ d.page_content for d in self.docs ]
        context = ". ".join(context_list)
        return context
    
        
    def LoadDatabase(self)-> bool : 
        try:
            
            
            self.vec_store = FAISS.load_local(
                        self.filePath,
                        self.embeddings,
                        allow_dangerous_deserialization=True
            )
            
            self.retriever=  self.vec_store.as_retriever(search_type='similarity', search_kwargs={"k":5})
            #Retrieve all documents for BM25Retriever
            #self.allDocuments= list( self.vec_store.docstore.__dict.values())
            self.allDocuments =           [
                self.vec_store.docstore.search(doc_id)
                for doc_id in self.vec_store.index_to_docstore_id.values() 
            ]
            
            self.bm25 = BM25Retriever.from_documents(self.allDocuments)
            self.bm25.k= 10 
            self.ensembel_retriever = EnsembleRetriever(retrievers=[ self.bm25, self.retriever], weights=[0.5, 0.5])
            # self.reranker_retriever = ContextualCompressionRetriever(
            #     base_compressor= 
            # )
            self.reranker_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever= self.ensembel_retriever
        )
            
            return True         
        except Exception as e:
            print(e)
            return False 

            