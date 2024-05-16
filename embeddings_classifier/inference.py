import time
import torch
from torch import nn
from embeddings_classifier.trainer import LinearClassifier
from embeddings_classifier.embeddings_generator import cohere_embeddings

# Load the trained model
model = LinearClassifier(in_features=1024, hidden_dim=512, num_classes=4)
model.load_state_dict(torch.load('embeddings_classifier/best_sentiment_classifier.pth'))
model.eval()  # Set the model to evaluation mode


def predict_sentiment(embeddings):
    with torch.no_grad():
        inputs = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


if __name__ == '__main__':
    text = "I hate this product, it's stupid and useless."
    new_embeddings = cohere_embeddings([text])
    sentiment_label = predict_sentiment(torch.tensor(new_embeddings[0]))
    print(f'Predicted sentiment label: {sentiment_label}')