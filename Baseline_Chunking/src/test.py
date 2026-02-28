from DataRetriever import DataRetriever
from RAG_Chunking import RAG_Chunking
from dotenv import load_dotenv
import os 

load_dotenv()

path=r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking\data\faiss_index"

dr = DataRetriever(path)

# if dr.LoadDatabase():
#     docs = dr.retriever.invoke('What was NVIDIA total revenue in 2024')
#     print(docs)


    
system_prompt ="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                If the answer is not found, say "I don't know."
"""
rc = RAG_Chunking('gpt-4o-mini', system_prompt)

question = 'What was NVIDIA total revenue in 2024?'

context= ''

if dr.LoadDatabase():
    docs = dr.retriever.invoke(question)
    context_list = [ d.page_content for d in docs ]
    context = ". ".join(context_list)
    
    print(context)


resp = rc.generate_answer_with_context(question, context )

print(resp)
