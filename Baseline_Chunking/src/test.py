from DataRetriever import DataRetriever
from RAG_Chunking import RAG_Chunking
from dotenv import load_dotenv
import os 
from RAGEvaluator import RAGEvaluator
from LLMEvaluator import LLMEvaluator
import json 


load_dotenv()

path=r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking\data\index_data\faiss_index_base_chunk_600"
       
dr = DataRetriever(path)


    
system_prompt ="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                If the answer is not found, say "I don't know."
"""
rc = RAG_Chunking('gpt-4o-mini', system_prompt)


 

questions = [

    # ---------- NVIDIA (Numeric / Financial) ----------

"What are some best practices outlined in the AWS Well-Architected Framework for optimizing storage costs and performance?",
"What were the net unrealized gains on investments in publicly-held equity securities for fiscal year 2025?",
"What would be the impact on income before taxes if there was a 10% adverse foreign exchange rate change?",
"What was the status of the commercial paper program as of January 25, 2026?",
"What caused the decrease in the effective tax rate for fiscal year 2023?",
"What are Money Market Funds in the context of eligible securities?",
"What was the revenue from the United States on Jan 28, 2024?",
"Can you tell me what SEC01-BP03 is and how it help in identify and validate control objectives in compliance?",
"What is the significance of the Amended and Restated Directorsâ€™ Indemnification Trust Agreement involving Microsoft Corporation?",
"How does Microsoft aim to address the evolving needs of customers through its technology solutions?",
"What is GAAP in United States of America?",
"What kind of things is the Federal National Mortgage Association involved with in terms of government securities?",
"How productivity relate to property and equipment in the financial report?",
"How did NVIDIA Corporation's revenue from specialized markets in 2026 contribute to overall productivity compared to the previous year?",
"What are the ways to prevent incidents according to the AWS Well-Architected Framework, and how can we improve alerts to reduce time-to-detection by 50%?",
"What is the role of the independent registered public accounting firm in relation to the consolidated financial statements of Microsoft Corporation?",
"How does NVIDIA contribute to productivity in business processes?",
"What were the earnings per share figures for the company in fiscal year 2025, and how did the company return value to shareholders through dividends during the same period?",
"What are the successes and failures associated with Chaos Engineering, and how do they relate to the principles of resilience engineering?",
"What was the Microsoft Cloud revenue in fiscal years 2025 compared to 2024?",
"What Microsoft ethics code apply to finance people and how it relate to their tech solutions?",
"How does the role of the Chief Financial Officer relate to the management's responsibility for internal control over financial reporting as specified in the SEC rules?",
"What are the implications of our operations in Ireland regarding income tax audits and how do they relate to our international properties?",
"What are the expected capital expenditures for fiscal year 2027 compared to fiscal year 2026, and how do these expenditures relate to the company's growth strategy?",
"What are the anticipated changes in capital expenditures for fiscal year 2027 compared to fiscal year 2026, and how do these changes relate to the company's future growth strategy?"


]
 
use_case={
        'id' : 'UC:1',
        'name' : 'Base chunk size',
        'llm_model' : 'gpt-4o-mini',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'retriever': {
            'chunk_size' : 600,
            'chunk_overlap': 60
        }
    }

use_case_id = use_case['id']


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

lEval = LLMEvaluator(rc.llm, dr.embeddings)

if dr.LoadDatabase():
    for q in questions:
        question = q 
        print(f'evaluating : {question}')
        context = dr.retrieve_context(question)
        ans = rc.generate_answer_with_context(question,context)
        
        #print(ans.content)
        
    
        
        # ca_score = lEval.compute_context_adherence(question,context,ans.content)        
        
        
        # cp_score = lEval.compute_context_precision(question, dr.docs)
        # # print(cp_score)
        
        # ans_rel = lEval.compute_answer_relevance(question, ans.content)
        # # print(ans_rel)
        
        # ground_score= lEval.compute_groundedness(context, ans.content)
        
        single_score= {
            'UseCase_id' : use_case_id ,
            'QNo' : no, 
            'question' : question,
            'context' : context,
            'answer' : ans.content,
            'context_adherence': "",
            'context_precision': "",
            'answer_relevance': "",
            'groundedness': ""
        }
        
        
        rag_scores.append(single_score)
        no+=1
        


save_list_to_file(rag_scores,'RAGAS_rag.json')


        
        
        
        
        
        
    
    
