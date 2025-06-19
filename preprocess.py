import os
import pandas as pd
from collections import Counter
import regex as re

df = pd.DataFrame(columns=["Prompt", "Model 1 - Mistral-7B", "Model 2 - MPT-7B", "Model 3 - StableLM-alpha-7B", "Ranking"])

def check_question_exists(row, model_name):
    prompt = row["Prompt"].lower()
    response = row[model_name].lower()

    return prompt in response

def delete_question(row, model_name):
    prompt = row["Prompt"].lower()
    response =  row[f"{model_name}_no_rep"].lower() # DELETE QUESTION FROM THE RESPONSE WITHOUT REPETITION SO BOTH ARE DELETED

    return response.replace(prompt, "")

def delete_repetition(response):
    response = response.lower()
    temp = (response + response).find(response, 1, -1) 
    if temp == -1: # No repetition
        return response 
    first_part = response[:temp]
    second_part = response[temp:]

    return first_part + second_part.replace(first_part, "")

def check_ranking(number_of_na, ranks):
    correct_ranking = None
    match number_of_na:
        case 0:# Probably there are no problems in this case
            correct_ranking = ranks == "1,2,3"
        case 1:
            correct_ranking = ranks == "0,1,2"
        case 2:
            correct_ranking = ranks == "0,0,1"
        case 3: # Probably there are no problems in this case
            correct_ranking = ranks == "0,0,0" 
        case _:
            correct_ranking = False
    return correct_ranking
        

def extract_preds(raw_pred):
    try:
        mistral = re.findall(r"[\*]*Content A[\*]*:[\*]* ([123\[\]N/ARank]+)", raw_pred)[0]
        mistral = mistral.replace("[", "").replace("]","").replace("Rank","").strip()
        mpt = re.findall(r"[\*]*Content B[\*]*:[\*]* ([123\[\]N/ARank]+)", raw_pred)[0]
        mpt = mpt.replace("[", "").replace("]","").replace("Rank","").strip()
        stable = re.findall(r"[\*]*Content C[\*]*:[\*]* ([123\[\]N/ARank]+)", raw_pred)[0]
        stable = stable.replace("[", "").replace("]","").replace("Rank","").strip()
        return f"{mistral}, {mpt}, {stable}"
    except:
        return "-1, -1, -1"

def rankings_LLM(ranking):
    ranking = ranking.rstrip(",")# sometimes they have , at the end
    ranks = [r.replace("N/A", "0").replace("NA","0").replace('"',"").replace("N/aA","0").replace("[","").replace("]","").strip() for r in ranking.split(",")]
    #we gotta make sure if there are N/A values in the ranking, the others should start from 1 not from 2 or 3
    number_of_na = ranks.count("0")
    
    ranked_ranks = ",".join(str(x) for x in sorted([int(rank) for rank in ranks]))
    correct_ranking = check_ranking(number_of_na, ranked_ranks)
    
    if not correct_ranking:# try to correct
        if ranked_ranks == "0,1,3": #FIRST EDGE CASE original ranking has 1,3 and N/A, so we make 3 2
            corrected_ranking = ["2" if int(rank)==3 else rank for rank in ranks]
        elif ranked_ranks == "0,0,3": #SECOND EDGE CASE original ranking has N/A, N/A, 3 so we make 3 1
            corrected_ranking = ["1" if int(rank)==3 else rank for rank in ranks]
        else: # regular case where we just reduce each ranking except 0(NA) by 1
            corrected_ranking = [str(int(rank) - 1) if int(rank)!=0 else rank for rank in ranks]
        number_of_na_in_corrected = corrected_ranking.count("0")
        ranked_ranks_new = ",".join(str(x) for x in sorted([int(rank) for rank in corrected_ranking]))
        correct_ranking_new = check_ranking(number_of_na, ranked_ranks_new)
        if not correct_ranking_new or number_of_na != number_of_na_in_corrected:
            return "-1, -1, -1"
        return ",".join(corrected_ranking)
    return ",".join(ranks)

def get_ranking_from_no_cot(ranking, prompt, prev_df):
    prev_df["prompt"] = prev_df["prompt"].apply(lambda x: x.lower())
    if "-1" not in ranking and "extracted" not in ranking:
        return ranking
    row = prev_df[prev_df["prompt"]==prompt.lower()]
    return row.extracted_pred.values[0]

def rankings(ranking):
    ranking = ranking.rstrip(",")# sometimes they have , at the end
    ranks = [r.replace("N/A", "0").replace("NA","0").replace('"',"").replace("N/aA","0").replace("[","").replace("]","").strip() for r in ranking.split(",")]
    #we gotta make sure if there are N/A values in the ranking, the others should start from 1 not from 2 or 3
    number_of_na = ranks.count("0")
    
    ranked_ranks = ",".join(str(x) for x in sorted([int(rank) for rank in ranks]))
    correct_ranking = check_ranking(number_of_na, ranked_ranks)
    
    if not correct_ranking:# try to correct
        if ranked_ranks == "0,1,3": #FIRST EDGE CASE original ranking has 1,3 and N/A, so we make 3 2
            corrected_ranking = ["2" if int(rank)==3 else rank for rank in ranks]
        elif ranked_ranks == "0,0,3": #SECOND EDGE CASE original ranking has N/A, N/A, 3 so we make 3 1
            corrected_ranking = ["1" if int(rank)==3 else rank for rank in ranks]
        else: # regular case where we just reduce each ranking except 0(NA) by 1
            corrected_ranking = [str(int(rank) - 1) if int(rank)!=0 else rank for rank in ranks]
        number_of_na_in_corrected = corrected_ranking.count("0")
        ranked_ranks_new = ",".join(str(x) for x in sorted([int(rank) for rank in corrected_ranking]))
        correct_ranking_new = check_ranking(number_of_na, ranked_ranks_new)
        if not correct_ranking_new or number_of_na != number_of_na_in_corrected:
            return ["-1", "-1", "-1"]
        return corrected_ranking
    return ranks

#https://www.geeksforgeeks.org/python-check-if-string-repeats-itself/
def check_repetition(response):
    response = response.lower()
    temp = (response + response).find(response, 1, -1) 
    if temp != -1: 
        return True
    return False

for group in range(1, 18):
    file_names = os.listdir(f"submissions/Group {group}")
    csv_file = None
    for file in file_names:
        if file.endswith(".csv"):
            csv_file = file
            break
    try:
        data = pd.read_csv(f"submissions/Group {group}/{csv_file}")
    except:
        data = pd.read_csv(f"submissions/Group {group}/{csv_file}", encoding='latin-1')
    if group == 10: # THEY USED 1 AS THE MOST HARMFUL AND 3 AS THE LEAST HARMFUL, SO REVERSE ORDER
        data["Ranking"] = data["Ranking"].apply(lambda x: ",".join(["N/A" if i=="N/A" else "3" if int(i)==1 else "1" if int(i)==3 else i for i in x.split(",")]))
    data = data.dropna(subset=["Model 1 - Mistral-7B", "Model 2 - MPT-7B", "Model 3 - StableLM-alpha-7B", "Ranking"])
    data["Group no"] = group
    df = pd.concat([df, data], ignore_index=True)


for model_name in ["Model 1 - Mistral-7B", "Model 2 - MPT-7B", "Model 3 - StableLM-alpha-7B"]:
    df[f"{model_name}_q_exists"] = df.apply(lambda row: check_question_exists(row, model_name), axis=1)
    df[f"{model_name}_repeats"] = df[model_name].apply(check_repetition)

    print(df[f"{model_name}_q_exists"].value_counts())
    print(df[f"{model_name}_repeats"].value_counts())
    df[f"{model_name}_no_rep"] = df[model_name].apply(delete_repetition)
    df[f"{model_name}_no_question"] = df.apply(lambda row: delete_question(row, model_name), axis=1)
df['Ranking'] = df['Ranking'].apply(lambda x: x.replace("NA","N/A").replace('"',"").replace("N/aA","N/A").replace("[","").replace("]","").replace(" ","").strip())
df["Mistral Rank"], df["MPT Rank"], df["Stable Rank"] = zip(*df['Ranking'].map(rankings))
df.to_csv("submissions/combined.csv", index=False)

gpt4o_no_cot = pd.read_csv("gpt4o_predictions.csv") 
gpt4o_no_cot["prompt"] = gpt4o_no_cot["prompt"].apply(lambda x: x.lower().strip())
gpt4o_no_cot.to_csv("gpt4o_predictions.csv", index=False) # one example becomes -1,-1,-1 because of a tie prediction

sonnet_no_cot = pd.read_csv("sonnet_predictions.csv") 
sonnet_no_cot["prompt"] = sonnet_no_cot["prompt"].apply(lambda x: x.lower().strip())
sonnet_no_cot.to_csv("sonnet_predictions.csv", index=False)

llama_no_cot = pd.read_csv("llama_predictions.csv") 
llama_no_cot["prompt"] = llama_no_cot["prompt"].apply(lambda x: x.lower().strip())
llama_no_cot.to_csv("llama_predictions.csv", index=False)# a few examples become -1,-1,-1 because of a tie prediction