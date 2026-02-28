from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_core.prompts import ChatPromptTemplate


class RAG_Chunking:
    system_prompt:str = ''
    sm:SystemMessage= None 
    llm : ChatOpenAI = None 
    llm_model :str =''
    
    def __init__(self,llm_model:str , systemprompt:str):
        self.system_prompt = systemprompt
        self.llm_model = llm_model
        
        if systemprompt is not None :
            self.sm= SystemMessage(systemprompt)
            
        if self.llm_model != '':
            self.llm = ChatOpenAI(
                model= self.llm_model,
                temperature=0.1,
                max_tokens=500,
                timeout= 2000,
                top_p=0.95 
            )
    
    def generate_answer(self,question:str)->AIMessage:
           hm = HumanMessage(question)
           
           chat_prompt= ChatPromptTemplate.from_messages(
               [
             self.sm,
             hm
               ]
        )
        
           chain = chat_prompt | self.llm
           response = chain.invoke({})
           return response                
                

    def generate_answer_with_context(self,question:str, context:str='')->AIMessage:
        
        
        
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt ),
            ("human", """
                Context:
                {context}

                Question:
                {question}
            """)
        ])
        
        
        response = None
        chain = chat_prompt | self.llm
        try:
            
            response = chain.invoke({
                "context" : context,
                "question": question
            })
        except Exception as e :
            response = None
        
        return response            
        
        
        

