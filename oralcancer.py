# -*- coding: utf-8 -*-
import os
import time
import copy
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print("drive.mount não executado (não Colab ou mount já feito).")

!pip install kaggle --quiet

from google.colab import files as gfiles
gfiles.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d zaidpy/oral-cancer-dataset
!unzip -q oral-cancer-dataset.zip -d data

data_dir = "./data/Oral cancer Dataset 2.0/OC Dataset kaggle new"
if not os.path.exists(data_dir):
    possible = [p for p in ["./data", "./dataset", "./datasets"] if os.path.exists(p)]
    if possible:
        data_dir = possible[0]
print("Usando data_dir =", data_dir)
print("Arquivos/dirs em data_dir:", os.listdir(data_dir) if os.path.exists(data_dir) else "NAO ENCONTRADO")

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Classes encontradas: {class_names}")
print(f"Total de imagens: {len(full_dataset)}")
print("Device:", device)

def calculate_metrics(model, dataloader):
    """Calcula precisão, recall e F1-score."""
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    precision = precision_score(labels_all, preds)
    recall = recall_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    return precision, recall, f1

def safe_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def export_probabilities_to_csv(model, dataloader, model_name, fold_path, split_name):
    model.eval()
    csv_probs  = os.path.join(fold_path, f"{model_name}_{split_name}.csv")
    csv_labels = os.path.join(fold_path, f"{split_name}_labels.csv")

    writer_probs  = csv.writer(open(csv_probs,  "w+", newline=""))
    writer_labels = csv.writer(open(csv_labels, "w+", newline=""))

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs   = torch.nn.functional.softmax(outputs, dim=1)

            batch_size = inputs.size(0)
            for i in range(batch_size):
                try:
                    if isinstance(dataloader.dataset, Subset):
                        img_idx = dataloader.dataset.indices[batch_idx * dataloader.batch_size + i]
                        img_path, _ = dataloader.dataset.dataset.samples[img_idx]
                    else:
                        img_path, _ = dataloader.dataset.samples[batch_idx * dataloader.batch_size + i]
                except Exception:
                    img_path = f"img_batch{batch_idx}_idx{i}"
                img_name = os.path.basename(img_path)

                prob = probs[i].cpu().numpy().tolist()
                writer_probs.writerow(prob + [img_name])
                writer_labels.writerow([img_name, int(labels[i].cpu().item())])


def plot_and_save_loss_acc(train_history, model_name, fold_path):
    # Loss
    plt.figure()
    plt.plot(train_history["train_loss"], label="Train Loss")
    plt.plot(train_history["val_loss"], label="Val Loss")
    plt.title(f"Loss per Epoch - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(fold_path, f"{model_name}_Loss.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(train_history["train_acc"], label="Train Acc")
    plt.plot(train_history["val_acc"], label="Val Acc")
    plt.title(f"Accuracy per Epoch - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(fold_path, f"{model_name}_Accuracy.png"))
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, model_name, fold_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(6,6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(fold_path, f"{model_name}_confusion_matrix.png"))
    plt.close()

def plot_and_save_roc(y_true, y_score, model_name, fold_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(fold_path, f"{model_name}_roc_curve.png"))
    plt.close()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    train_history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                train_history["train_acc"].append(epoch_acc.item())
                train_history["train_loss"].append(epoch_loss)
            else:
                train_history["val_acc"].append(epoch_acc.item())
                train_history["val_loss"].append(epoch_loss)

            print(f"{phase} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_acc.item(), train_history

models_to_run = ["googlenet", "resnet18", "densenet121"]
batch_size = 32
num_epochs = 40
lr = 1.5e-5
step_size = 10
gamma = 0.1
n_splits = 5

# Pasta base no Drive
base_save_dir = "/content/drive/MyDrive/OralCancer"
safe_makedirs(base_save_dir)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_indices = list(kf.split(np.arange(len(full_dataset))))

for model_name in models_to_run:
    print("\n" + "="*80)
    print(f"INICIANDO treinos para modelo: {model_name}")
    print("="*80)

    model_base_dir = os.path.join(base_save_dir, model_name)
    safe_makedirs(model_base_dir)

    fold_accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for fold, (train_idx, val_idx) in enumerate(all_indices):
        print(f"\n===== {model_name.upper()} - FOLD {fold+1}/{n_splits} =====")
        fold_start_time = time.time()

        fold_path = os.path.join(model_base_dir, f"fold_{fold+1}")
        safe_makedirs(fold_path)

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        dataloaders = {
            "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            "val":   DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }
        dataset_sizes = {"train": len(train_subset), "val": len(val_subset)}

        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name == "densenet121":
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        elif model_name == "googlenet":
            model = models.googlenet(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        model, best_acc, train_history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=num_epochs)
        fold_accuracies.append(best_acc)

        precision, recall, f1 = calculate_metrics(model, dataloaders["val"])
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        export_probabilities_to_csv(model, dataloaders["train"], model_name, fold_path, "train")
        export_probabilities_to_csv(model, dataloaders["val"],   model_name, fold_path, "test")

        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        plot_and_save_confusion_matrix(all_labels, all_preds, model_name, fold_path)

        test_csv = os.path.join(fold_path, f"{model_name}_test.csv")
        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv, header=None)
            probs_arr = np.asarray(df)[:, :-1].astype(float)
            if probs_arr.shape[1] >= 2:
                y_score = probs_arr[:, 1]
            else:
                y_score = probs_arr[:, 0]
            labels_csv = os.path.join(fold_path, "test_labels.csv")
            if os.path.exists(labels_csv):
                df_lbl = pd.read_csv(labels_csv, header=None)
                y_true = np.asarray(df_lbl)[:, 1].astype(int)
            else:
                y_true = np.array(all_labels)
        else:
            y_true = np.array(all_labels)
            y_score = np.array(all_preds)

        plot_and_save_roc(y_true, y_score, model_name, fold_path)

        try:
            plot_and_save_loss_acc(train_history, model_name, fold_path)
        except Exception as e:
            print("Erro ao gerar Loss/Acc plots:", e)

        fold_time = (time.time() - fold_start_time) / 60.0
        print(f"Tempo do fold {fold+1}: {fold_time:.2f} minutos")
        print(f"Arquivos salvos em: {fold_path}")

    print("\n===== RESULTADOS FINAIS —", model_name.upper(), "=====")
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Acurácia média (5-fold): {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    summary_csv = os.path.join(model_base_dir, f"{model_name}_folds_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_val_acc", "precision", "recall", "f1"])
        for i in range(len(fold_accuracies)):
            writer.writerow([i+1, fold_accuracies[i], precisions[i], recalls[i], f1s[i]])
    print(f"Resumo por folds salvo em: {summary_csv}")

print("\n\n=== EXECUÇÃO COMPLETA ===")
print("Resultados salvos em:", base_save_dir)