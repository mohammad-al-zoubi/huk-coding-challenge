inference_times = {
    'gpt': 0.9,
    'claude': 1.14,
    'llama': 2.0,
    'linear classifier': 0.001
}

micro_f1_scores = {
    'gpt': 0.49,
    'claude': 0.46,
    'llama': 0.48,
    'linear classifier': 0.64
}

macro_f1_scores = {
    'gpt': 0.4,
    'claude': 0.42,
    'llama': 0.39,
    'linear classifier': 0.61
}

gpt_class_f1_scores = {
    'positive': {'f1': 0.61},
    'neutral': {'f1': 0.11},
    'irrelevant': {'f1': 0.19},
    'negative': {'f1': 0.67}
}

claude_class_f1_scores = {
    'positive': {'f1': 0.60},
    'neutral': {'f1': 0.22},
    'irrelevant': {'f1': 0.21},
    'negative': {'f1': 0.64}
}

llama_class_f1_scores = {
    'positive': {'f1': 0.59},
    'neutral': {'f1': 0.22},
    'irrelevant': {'f1': 0.11},
    'negative': {'f1': 0.64}
}

linear_class_f1_scores = {
    'positive': {'f1': 0.67},
    'neutral': {'f1': 0.59},
    'irrelevant': {'f1': 0.44},
    'negative': {'f1': 0.67}
}
