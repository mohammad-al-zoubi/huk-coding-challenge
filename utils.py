import json

import yaml

CLASSES_DICT = {
    0: 'Negative',
    1: 'Irrelevant',
    2: 'Neutral',
    3: 'Positive'
}


def get_f1_score():
    ...


def get_average_inference_time():
    ...


def load_yaml_to_dict(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def load_json_to_dict(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
