import json
from pathlib import Path
from logging import getLogger

import tqdm
import openai
import anthropic
from groq import Groq

import pandas as pd
from configuration import config
from utils import load_json_to_dict

logger = getLogger(__name__)

PROMPT = """
Sentiment Classification Task:

You are an expert sentiment analysis model trained to classify text into one of four sentiment categories: positive, negative, neutral or irrelevant. Your task is to carefully analyze the given text and provide the corresponding sentiment label as an output, following the specified format.

Instructions:

1. Read the given text carefully and analyze its sentiment.
2. Determine whether the sentiment expressed in the text is positive, negative, neutral or irrelevant.
3. Output your classification result as a JSON object with the key 'result' and the corresponding value based on the following mapping:
   - If the sentiment is positive, output: {"result": 3}
   - If the sentiment is neutral, output: {"result": 2}
   - If the sentiment is irrelevant, output: {"result": 1}
   - If the sentiment is negative, output: {"result": 0}

Example:
If the text expresses a negative sentiment, your output should be:
{"result": 0}

Please provide your output strictly in the specified JSON format. Do not include any additional text or explanations.

Text to classify:\n\n
"""

ANTHROPIC_API_KEY = config['llm']['ANTHROPIC_API_KEY']
claude_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=ANTHROPIC_API_KEY,
)
openai.api_key = config['llm']['OPENAI_API_KEY']
groq_client = Groq(api_key=config['llm']['GROQ_API_KEY'])


def get_claude_response(prompt):
    message = claude_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']


def get_llama_response(prompt):
    chat_completion = groq_client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content


def load_test_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    new_cols = {df.columns[0]: 'id', df.columns[1]: 'name', df.columns[2]: 'rating', df.columns[3]: 'text'}
    df = df.rename(columns=new_cols)
    return df


def predict_sentiment(text, mode='claude'):
    if mode == 'claude':
        response = get_claude_response(PROMPT + text)
    elif mode == 'gpt':
        response = get_gpt_response(PROMPT + text)
    elif mode == 'llama':
        response = get_llama_response(PROMPT + text)
    else:
        raise ValueError("Invalid model name. Please choose either 'claude' or 'gpt'.")

    result = json.loads(response)['result']
    return result


def evaluate_model(data, mode='llama'):
    logger.info(f"Predicting sentiment using {mode} model...")
    results_path = Path('llm') / f'{mode}_results.json'
    try:
        results = load_json_to_dict(results_path)
    except FileNotFoundError:
        results = []
    # results = []
    condition = False
    for i, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        text = row['text']
        if row['id'] == 5832:
            condition = True
        if condition:
            result = predict_sentiment(text, mode)
            results.append({'id': row['id'], 'result': result})
            with open(results_path, 'w', encoding='utf8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    data = load_test_data(config['general']['path_to_val_csv'])
    results = evaluate_model(data, mode='llama')
    print()
