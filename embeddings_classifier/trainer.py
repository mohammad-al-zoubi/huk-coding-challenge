import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer


class LinearClassifier(nn.Module):
    """
    A simple linear classifier with two linear layers and a softmax activation.
    It classifies embeddings of size `in_features` into `num_classes` classes.
    """

    def __init__(self, in_features, hidden_dim, num_classes=4):
        super(LinearClassifier, self).__init__()
        self.linear_1 = nn.Linear(in_features, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_2(out)
        out = self.softmax(out)
        return out


# TODO: Implement
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: Labels should be class indices
        embeddings, labels = self.data[index]
        # Process your data here if needed
        return embeddings, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_train_data = 'path/to/train/data'
path_to_eval_data = 'path/to/eval/data'

# Instantiate your model, dataset, and dataloader
net = LinearClassifier(in_features=768, hidden_dim=64, num_classes=4)
train_dataset = SentimentDataset(path_to_train_data)
eval_dataset = SentimentDataset(path_to_eval_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)


optimizer = AdamW(net.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

torch.save(net.state_dict(), 'embeddings_classifier/sentiment_classifier.pth')
print('Finished Training')
