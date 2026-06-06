from build_eval_data import LoadEvalData

from ragas.dataset import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import os 
import pandas as pd 

eval_dataset = LoadEvalData()

print(eval_dataset)

api_key = os.getenv("OPENAI_API_KEY")
base_url=os.getenv("OPENAI_API_BASE")

     

evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        base_url=base_url
    )
)

# response = evaluator_llm.langchain_llm.invoke(
#     "What is 2 + 2?"
# )

# print(response.content)

#Import metrics 

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

response = evaluator_llm.langchain_llm.invoke(
    "Reply with SUCCESS"
)

print(type(evaluator_llm))
print(type(evaluator_llm.langchain_llm))



#Run evaluation
from ragas import evaluate

from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# print('running evaluation')



result = evaluate(
    dataset=eval_dataset,
    metrics=[
        
         
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
         
    ],
    llm=evaluator_llm,
    embeddings=embeddings
)

df_results = result.to_pandas()
df_results.to_csv("RAGAS_Eval_Result.csv")
print('ragas evaluation completed')

