from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class LLMEvaluator:
    _embedding:HuggingFaceEmbeddings = None 
    ca_system = """
    You are an expert AI evaluator.

Your task is to evaluate CONTEXT ADHERENCE.

Definition:
Context Adherence measures whether the generated answer is strictly supported by the provided context.

Instructions:
- Compare the Answer against the Context.
- Identify if any part of the answer introduces information NOT present in the context.
- If all claims are fully supported → score = 1.0
- If partially supported → score between 0.0 and 1.0
- If mostly unsupported or hallucinated → score = 0.0

Output:
strictly value between 0 to 1 
    """
    
    context_precision_system_prompt="""
    You are an expert AI evaluator.

Your task is to evaluate CONTEXT PRECISION.

Definition:
Context Precision measures how relevant the retrieved document chunk is to the question.

Instructions:
- Read the Question carefully.
- Evaluate whether the Document is directly useful in answering the Question.
- If highly relevant → score = 1.0
- If partially relevant → score between 0.0 and 1.0
- If irrelevant → score = 0.0

Do NOT evaluate answer correctness.
Only evaluate relevance of the document to the question.

Output:
strictly value between 0 to 1 
    """
    
    def __init__(self,llm,embedding):
        self._llm = llm 
        self._embedding = embedding
        
        
    def compute_context_precision(self,question, retrieved_docs):
        relevant = 0
         
        
        for doc in retrieved_docs:
            prompt = f"""
            Question: {question}

            Document:
            {doc.page_content}

            Is this document relevant to the question?
            Answer Yes or No.
            
            Output:
            strictly value Yes or No
            """
            response = self._llm.invoke(prompt)

            if "yes" in response.content.lower():
                relevant += 1

        return relevant / len(retrieved_docs)
    

    def compute_answer_relevance(self, question, answer):

        q_vec = self._embedding.embed_query(question)
        a_vec = self._embedding.embed_query(answer)

        similarity = cosine_similarity(
            [q_vec], [a_vec]
        )[0][0]

        return float(similarity)

    def compute_context_adherence(self, iquestion, icontext, ianswer) -> float :

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", self.ca_system 
                ),
                (
                    "human", """
                    Question: {question}

                    Context:
                    {context}

                    Answer:
                    {answer}
                    """
                )
            ]
        )

        
        result = None 
        chain = chat_prompt | self._llm  
        
        try:
            response = chain.invoke(
            {
                 "question" : iquestion ,
                 "context" : icontext,                 
                 "answer" : ianswer 
            }
        )
            result = float(response.content.strip())
        except Exception as e:
            print(e )
           
            
         
        
        return result