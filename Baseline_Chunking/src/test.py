from DataRetriever import DataRetriever
from RAG_Chunking import RAG_Chunking
from dotenv import load_dotenv
import os 
from RAGEvaluator import RAGEvaluator
from LLMEvaluator import LLMEvaluator
import json 


load_dotenv()

path=r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking\data\faiss_index"

dr = DataRetriever(path)


    
system_prompt ="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                If the answer is not found, say "I don't know."
"""
rc = RAG_Chunking('gpt-4o-mini', system_prompt)


 

questions = [

    # ---------- NVIDIA (Numeric / Financial) ----------

    "What was NVIDIA total revenue for fiscal year 2024?",
    "What was NVIDIA Data Center revenue in fiscal year 2024?",
    "How much operating income did NVIDIA report in fiscal year 2024?",
    "What percentage of NVIDIA revenue came from customers headquartered outside the United States in fiscal year 2024?",
    "What were NVIDIA total long-lived assets as of January 25, 2026?",

    # ---------- NVIDIA (Segment / Lease / Financial Detail) ----------

    "What segments does NVIDIA report in its segment information?",
    "What does NVIDIA include in its Compute & Networking segment?",
    "How much depreciation expense did NVIDIA report in fiscal year 2024?",
    "What are NVIDIA future operating lease obligations for fiscal year 2027?",
    "How does NVIDIA define direct customers versus indirect customers?",

    # ---------- Microsoft (Risk / Competition / Cloud) ----------

    "What strategic and competitive risks does Microsoft disclose in its risk factors?",
    "How does Microsoft describe competition in the technology sector?",
    "What products are included in Microsoft Intelligent Cloud segment?",
    "How is Azure revenue primarily generated?",
    "What competitive advantages does Microsoft claim for Azure?",

    # ---------- Microsoft (Product / AI / Services) ----------

    "What services are included in Microsoft Enterprise and Partner Services?",
    "How does Microsoft describe its Azure AI offerings?",
    "What drives Dynamics revenue according to Microsoft?",
    "What factors impact Windows OEM revenue?",

    # ---------- AWS Well-Architected Framework ----------

    "What are the six pillars of the AWS Well-Architected Framework?",
    "What are the design principles of the Operational Excellence pillar?"

]
 
use_case={
        'id' : 'UC:1',
        'name' : 'default RAG',
        'llm_model' : 'gpt-4o-mini',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'retriever': {
            'chunk_size' : 1000,
            'chunk_overlap': 200
        }
    }

rag_scores=list()
def save_list_to_file(data_list, filename):
    """
    Saves a Python list to a file in JSON format.
    """
    if not isinstance(data_list, list):
        raise TypeError("data_list must be a list.")

    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data_list, file, ensure_ascii=False, indent=4)
        print(f"List saved successfully to '{filename}'.")
    except (OSError, IOError) as e:
        print(f"Error saving list to file: {e}")

no=1    

if dr.LoadDatabase():
    for q in questions:
        question = q 
        print(f'evaluating : {question}')
        context = dr.retrieve_context(question)
        ans = rc.generate_answer_with_context(question,context)
        
        #print(ans.content)
        
    
        lEval = LLMEvaluator(rc.llm, dr.embeddings)
        ca_score = lEval.compute_context_adherence(question,context,ans.content)        
        
        
        cp_score = lEval.compute_context_precision(question, dr.docs)
        # print(cp_score)
        
        ans_rel = lEval.compute_answer_relevance(question, ans.content)
        # print(ans_rel)
        
        ground_score= lEval.compute_groundedness(context, ans.content)
        
        single_score= {
            'UseCase_id' : 'UC:1',
            'QNo' : no, 
            'question' : question,
            'context' : context,
            'answer' : ans.content,
            'context_adherence': str(ca_score),
            'context_precision': str(cp_score),
            'answer_relevance': f"{ans_rel:.3f}",
            'groundedness': ground_score
        }
        
        
        rag_scores.append(single_score)
        no+=1
        


save_list_to_file(rag_scores,'regular_rag')


        
        
        
        
        
        
    
    
