import requests

def call_agent(question: str) -> str:
    try:
        res = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
        if res.status_code == 200:
            return res.json()["answer"]
        else:
            return "None"
    except Exception as e:
        print("Error while calling agentic backend:", e)
        return "None"

import re

def extract_answer(answer_text: str, option_map: dict = None) -> str:
    # 1. Try to find direct mention of option letter (A, B, C, D)
    match = re.search(r'(?:final answer|answer is|answer)\s*[:\-]?\s*\(?([A-D])\)?', answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Try to find boxed final value
    match = re.search(r'\\boxed{([^}]+)}', answer_text)
    if match and option_map:
        boxed = match.group(1).strip().replace(' ', '')
        for key, val in option_map.items():
            if boxed == val.replace(' ', ''):
                return key

    # 3. Try to match raw final value (number or expression)
    if option_map:
        # Check for final expression mentioned directly
        for key, val in option_map.items():
            val_clean = val.replace(' ', '')
            if val_clean in answer_text.replace(' ', ''):
                return key

    return "None"


