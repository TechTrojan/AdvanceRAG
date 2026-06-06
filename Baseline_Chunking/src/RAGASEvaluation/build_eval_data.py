import pandas as pd
import json 
from datasets import Dataset 

# #RAGAS dataset 
# df_ragas = pd.read_csv("testset_3.csv", names=[ "user_input","reference_contexts","reference","persona_name","query_style","query_length","synthesizer_name"], 
# sep=",")


# print(df_ragas.head())

# #LLM Result
# df_llm = pd.DataFrame()

# with open("RAGAS_RAG.JSON","r", encoding="utf-8") as f:
#     data = json.load(f)
    
# df_llm = pd.DataFrame(data)

# print(df_llm.head() )

# df_llm.to_csv("LLM_Result.csv") 

# from datasets import Dataset 
# import pandas as pd 

EVAL_DATA_SET="testset_3.CSV" 

# Load CSV
df = pd.read_csv(EVAL_DATA_SET)

# Build Dataset expected by RAGAS
eval_dataset = Dataset.from_dict({
    "user_input": df["user_input"].tolist(),
    "response": df["response"].tolist(),
    "retrieved_contexts": df["retrieved_contexts"].tolist(),
    "reference": df["reference"].tolist(),
})

print(eval_dataset)