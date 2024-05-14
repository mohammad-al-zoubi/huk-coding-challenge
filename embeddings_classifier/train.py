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
        embeddings, labels = self.data[index]
        # Process your data here if needed
        return embeddings, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate your model, dataset, and dataloader
net = LinearClassifier(in_features=768, hidden_dim=64, num_classes=4)
train_dataset = SentimentDataset(path)
eval_dataset = SentimentDataset(path)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

def hf_trainer():
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=10,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        logging_dir='./logs',  # directory for storing logs
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # eval dataset
    )

    # Start the training
    trainer.train()


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
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
