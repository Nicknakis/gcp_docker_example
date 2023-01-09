import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


@click.group()
def cli():
    pass

#gss
@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)
    losses = []
    # TODO: Implement training loop here
    images = torch.load("./data/processed/train_images")
    labels = torch.load("./data/processed/train_labs")

    model = MyAwesomeModel()
    train_set = torch.utils.data.TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 10
    for e in range(epochs):
        running_loss = 0
        acc = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.unsqueeze(1)
            # TODO: Training pass
            output = model(images)

            loss = criterion(output, labels)
            optimizer.zero_grad()  # clear the gradients.
            loss.backward()  # backpropagate
            optimizer.step()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            running_loss += loss.item()
            acc += accuracy.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            print(f"TRAIN Accuracy: {(acc/len(trainloader))*100}%")
            losses.append(running_loss / len(trainloader))

            torch.save(
                model.state_dict(),
                "trained_model.pt",
            )
    plt.figure()
    plt.plot(losses)
    plt.savefig("loss.pdf")
    plt.show()


@click.command()
@click.argument("model_checkpoint")
@click.argument("train_images")
@click.argument("train_labels")
def evaluate(model_checkpoint,train_images,train_labels):
    print("Evaluating until hitting the ceiling")
    print(train_images)
    print(model_checkpoint)
    images = torch.load(train_images)
    labels = torch.load(train_labels)
    preds, target = [], []
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    

    train_set = torch.utils.data.TensorDataset(images, labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.unsqueeze(1)
        # TODO: Training pass
        output = model(images)

        probs = torch.exp(output)
        preds.append(probs.argmax(dim=-1))
        target.append(labels.detach())
    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)
    report = classification_report(target, preds)
    with open("classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
    disp.plot()
    plt.savefig('confusion_matrix.png')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
