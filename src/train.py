import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

import mlflow
import mlflow.pytorch

from dataset import get_dataloaders
from model import build_model

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("cat-dog-classifier")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    num_epochs = 5
    learning_rate = 1e-3
    batch_size = 32
    freeze_backbone = True

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=batch_size
    )
    print(f"Classes : {class_names}")  # ex: ['cat', 'dog']

    model = build_model(
        num_classes=len(class_names), freeze_backbone=freeze_backbone
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "resnet18",
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "freeze_backbone": freeze_backbone,
            }
        )

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        # évaluation finale sur le test set
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Test accuracy: {test_acc:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        # sauvegarde du modèle
        mlflow.pytorch.log_model(
            model, "model", registered_model_name="cat-dog-classifier"
        )
        torch.save(model.state_dict(), "models/model.pt")

        mlflow.log_dict({"class_names": class_names}, "class_names.json")

        with open("models/class_names.json", "w") as f:
            json.dump({"class_names": class_names}, f)


if __name__ == "__main__":
    main()
