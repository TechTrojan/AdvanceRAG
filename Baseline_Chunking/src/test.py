from DataRetriever import DataRetriever
from RAG_Chunking import RAG_Chunking
from dotenv import load_dotenv
import os 

load_dotenv()

path=r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking\data\faiss_index"

# dr = DataRetriever(path)

# if dr.LoadDatabase():
#     docs = dr.retriever.invoke('What was NVIDIA total revenue in 2024')
#     print(docs)


    
system_prompt ='You are helpful assistant to answer questions.'
rc = RAG_Chunking('gpt-4o-mini', system_prompt)


resp = rc.generate_answer('What was NVIDIA total revenue in 2024?')

print(resp)
