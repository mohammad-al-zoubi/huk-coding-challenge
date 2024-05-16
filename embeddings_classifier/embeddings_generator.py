import json

import cohere
from utils import load_data
from configuration import config

co = cohere.Client(config['embeddings_classifier']['COHERE_API_KEY'])


def cohere_embeddings(passages, input_type="classification"):
    embeds = co.embed(texts=passages, model=config['embeddings_classifier']['cohere_embeddings_model'],
                      input_type=input_type).embeddings
    return embeds


def load_csv_to_dict(csv_file_path):
    data = load_data(csv_file_path)
    data_dicts = []
    for i, row in data.iterrows():
        data_dict = {'id': row['id'], 'text': str(row['text'])}
        data_dicts.append(data_dict)
    return data_dicts


def add_embeddings_to_dict(data_dicts, embeddings_list):
    for i, data_dict in enumerate(data_dicts):
        data_dict['embeddings'] = embeddings_list[i]
    return data_dicts


if __name__ == "__main__":
    csv_file_path = 'data/training.csv'
    passages_dict = load_csv_to_dict(csv_file_path)[:20000]
    passages = [item['text'] for item in passages_dict]
    embeddings = cohere_embeddings(passages)
    passages_dict = add_embeddings_to_dict(passages_dict, embeddings)
    with open('embeddings_training.json', 'w', encoding='utf-8') as file:
        json.dump(passages_dict, file, ensure_ascii=False)
