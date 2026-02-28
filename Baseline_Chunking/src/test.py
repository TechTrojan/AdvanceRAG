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

# if dr.LoadDatabase():

#     question = 'What was NVIDIA total revenue in 2024?'

#     context= dr.retrieve_context(question)
#     ans = rc.generate_answer_with_context(question,context)

#     print(ans.content)

 
samples = [

    {
        "question": "What was NVIDIA total revenue in 2024?",
        "ground_truth": "NVIDIA total revenue for fiscal year ended January 28, 2024 was $60,922 million.",
        "relevant_doc_ids": ["nvidia_10k_doc"]
    },

    {
        "question": "What risks does Microsoft describe regarding competition?",
        "ground_truth": "Microsoft faces intense competition across all markets from diversified global companies, specialized firms, open source offerings, and rapidly evolving technologies. Failure to innovate or deliver appealing products could adversely affect its business and results of operations.",
        "relevant_doc_ids": ["microsoft_10k_doc"]
    },

    {
        "question": "What are the six pillars of the AWS Well-Architected Framework?",
        "ground_truth": "The six pillars are Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, and Sustainability.",
        "relevant_doc_ids": ["aws_well_architected_doc"]
    },

    {
        "question": "What is the purpose of the Reliability pillar in AWS Well-Architected?",
        "ground_truth": "The Reliability pillar focuses on designing systems that deliver stable and efficient performance by ensuring workloads can meet expectations and requirements even under changing conditions.",
        "relevant_doc_ids": ["aws_well_architected_doc"]
    },

    {
        "question": "How do NVIDIA and Microsoft describe their AI offerings?",
        "ground_truth": "NVIDIA describes its AI solutions within its Compute & Networking segment, including data center accelerated computing and AI platforms. Microsoft describes Azure AI offerings as providing supercomputing power for AI at scale, complemented by cloud AI services, custom-built silicon, and developer platforms such as Azure AI Foundry.",
        "relevant_doc_ids": ["nvidia_10k_doc", "microsoft_10k_doc"]
    }

]


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

    

if dr.LoadDatabase():
    for q in samples:
        question = q["question"]
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
            'question' : question,
            'context' : context,
            'answer' : ans.content,
            'context_adherence': str(ca_score),
            'context_precision': str(cp_score),
            'answer_relevance': f"{ans_rel:.3f}",
            'groundedness': ground_score
        }
        
        rag_scores.append(single_score)
        


save_list_to_file(rag_scores,'regular_rag')


        
        
        
        
        
        
    
    
