import json
from pandas import pandas as pd , json_normalize   

def generate_csv_from_json(filePath: str):
    
    with open(filePath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔥 Fix: Convert groundedness string → dict
    for item in data:
        if isinstance(item.get("groundedness"), str):
            try:
                item["groundedness"] = json.loads(item["groundedness"])
            except:
                item["groundedness"] = {}

    # Normalize nested fields
    df = json_normalize(data)

    df.to_csv("eval_result_2.csv", index=False)

    print("CSV generated successfully!")
        


#save_list_to_file(rag_scores,'regular_rag')
filePath = ""
generate_csv_from_json( "regular_rag.json")