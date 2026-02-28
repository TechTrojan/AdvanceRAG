from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate


class RAGEvaluator:
    system_prompt:str = ''
    
    def __init__(self, retriever, llm, embeddings, systemprompt:str):
        self.retriever = retriever
        self.llm = llm
        self.embeddings = embeddings
        self.system_prompt = systemprompt
        
    """Precision@K = Relevant retrieved docs / K"""
    def compute_precision(self, retrieved_docs, sample, k=5):
        relevant_ids = set(sample["relevant_doc_ids"])

        retrieved_ids = [
            doc.metadata.get("doc_id")
            for doc in retrieved_docs[:k]
                ]

        relevant_retrieved = len(
            [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
        )

        return relevant_retrieved / k
    
    
    """Recall@K = Relevant retrieved docs / Total relevant docs"""
    def compute_recall(self, retrieved_docs, sample, k=5):
        relevant_ids = set(sample["relevant_doc_ids"])
        total_relevant = len(relevant_ids)

        retrieved_ids = [
            doc.metadata.get("doc_id")
            for doc in retrieved_docs[:k]
        ]

        relevant_retrieved = len(
            [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]
        )

        if total_relevant == 0:
            return 0

        return relevant_retrieved / total_relevant
    
    

    def compute_answer_similarity(self, generated_answer, sample):
        ground_truth = sample["ground_truth"]

        gen_emb = self.embeddings.embed_query(generated_answer)
        gt_emb = self.embeddings.embed_query(ground_truth)

        similarity = cosine_similarity([gen_emb], [gt_emb])[0][0]

        return float(similarity) 
    
    def evaluate_question(self, sample):
        question = sample["question"]

        # Retrieve
        
        docs = self.retriever.invoke(question)
        # context_list = [ d.page_content for d in docs ]
        # context = ". ".join(context_list)
        
        
        # chat_prompt = ChatPromptTemplate.from_messages([
        #     ("system", self.system_prompt ),
        #     ("human", """
        #         Context:
        #         {context}

        #         Question:
        #         {question}
        #     """)
        # ])
        # chain = chat_prompt | self.llm
         
        # # Generate
        
        # try:
            
        #     response = chain.invoke({
        #         "context" : context,
        #         "question": question
        #     })
        #     answer = response.content 
        # except Exception as e :
        #     response = None

        # Compute metrics
        precision = self.compute_precision(docs, sample)
        recall = self.compute_recall(docs, sample)
        
        
        return {
            "precision": precision,
            "recall": recall
        }