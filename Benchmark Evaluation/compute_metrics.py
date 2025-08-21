import numpy as np
import json
import pandas as pd

QUES_TYPES = ['MCQ', 'MCQ(multiple)', 'Integer', 'Numeric']
model = "Agentic_CoT"
response_file = "responses/Agentic_CoT_responses/responses.json"

def compute_score(gold, resp, question_type):
    if question_type == 'MCQ(multiple)':
        gold = set([c for c in 'ABCD' if c in gold])
        resp = set([c for c in 'ABCD' if c in resp])
        if resp == gold:
            return 1.0
        elif len(resp - gold) == 0:
            return 0.25 * len(resp)
        return 0.0
    elif question_type == 'MCQ':
        return int(set(gold) == set(resp))
    else:
        if resp == "None":
            return 0.0
        try:
            return int(abs(float(gold) - float(resp)) <= 0.01)
        except:
            return 0.0

def construct_table():
    dataset = json.load(open('data/dataset.json'))
    responses = json.load(open(response_file, encoding="utf-8"))

    rows = []
    indexed_responses = {(resp["description"], resp["index"]): resp for resp in responses}

    for q in dataset:
        key = (q['description'], q['index'])
        if key not in indexed_responses:
            print(f"⚠️ No response found for {key}, skipping.")
            continue

        resp_data = indexed_responses[key]
        qtype = q['type']
        gold = q['gold']

        try:
            response_entry = resp_data[f"{model}_response"]
            extract = response_entry['choices'][0].get('extract', "None")
        except KeyError:
            print(f"⚠️ Missing response format for {key}, skipping.")
            extract = "None"

        score = compute_score(gold, extract, qtype)

        rows.append({
            "Index": q['index'],
            "Description": q['description'],
            "Type": qtype,
            "Subject": q['subject'],
            "Gold": gold,
            "Agentic": extract,
            "Score": score
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/agentic_scores.csv", index=False)
    return df


df = construct_table()

# Aggregate results
agg_overall = df['Score'].mean()
agg_by_type = df.groupby('Type')['Score'].agg(['count', 'mean']).reset_index()
agg_by_subject = df.groupby('Subject')['Score'].agg(['count', 'mean']).reset_index()

agg_by_type.to_csv("results/agentic_aggregated_type.csv", index=False)
agg_by_subject.to_csv("results/agentic_aggregated_subject.csv", index=False)

print(f"✅ Overall Accuracy: {agg_overall:.3f}")
print("✅ Saved: agentic_scores.csv, agentic_aggregated_type.csv, agentic_aggregated_subject.csv")
