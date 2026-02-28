from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain_core.documents import Document


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
- If highly relevant → 1
- If irrelevant → score = 0.0

Do NOT evaluate answer correctness.
Only evaluate relevance of the document to the question.

Output:
strictly value between 0 to 1 
    """
    
    def __init__(self,llm,embedding):
        self._llm = llm 
        self._embedding = embedding
        
        
    def compute_context_precision(self,question:str, retrieved_docs:list[Document])->float:
        relevant = 0
        
        if len(retrieved_docs)==0 :
            return 0 
        
        chat_prompt= ChatPromptTemplate.from_messages(
            [
                ("system", self.context_precision_system_prompt) ,
                ("human", """
                    Question: 
                    {question}

                    Document:
                    {page_content}
                 """)
            ]
        )
        
        print(chat_prompt.input_variables)
        
        for doc in retrieved_docs:
            response = None 
            
            chain =  chat_prompt | self._llm
            
            try:
                response = chain.invoke(
                    {
                        "question": question,
                        "page_content": doc.page_content
                    }
                )

                if "1" in response.content:
                    relevant += 1
            except Exception as e :
                print(e) 
            

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