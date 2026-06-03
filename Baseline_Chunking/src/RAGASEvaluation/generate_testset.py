
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from pathlib import Path
import sys
from openai import OpenAI
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings



from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add src directory to Python path
#print(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from DataRetriever import DataRetriever


path=r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking\data\index_data\faiss_index_base_chunk_600"
       
dr = DataRetriever(path)
docs = None 
if dr.LoadDatabase():    
    docs = dr.RetriveAllDocs()
    if not docs  or len(docs) == 0 :
        print('No documents loaded')
    else:
        print(f'documents loaded {len(docs)}')



from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)


splitter = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 60
            
        )

chunks = splitter.split_documents(docs)
chunks_sample = chunks[:200]  # start with 200, increase if needed


generator = TestsetGenerator(llm=llm, embedding_model= embeddings)
import asyncio
import nest_asyncio
nest_asyncio.apply()

async def generate_testset():
    # Use synchronous method, not await
    testset = generator.generate_with_langchain_docs(
        chunks_sample,
        testset_size=10,
        query_distribution=[
            (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
        ]
    )
    return testset

testset = asyncio.run(generate_testset())
 


testset.to_pandas().to_csv("testset.csv", index=False) 

