from tkinter import ON
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

import os
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

n_epoch = 15
# n_split = 

# Load data from *.csv file
social_data = pd.read_csv("Data-Labeling.csv")
# Sort dataframe to fit with id's order
social_data = social_data.sort_values(by=['id'])
# print(social_data.head())

# Statistic number of each category
# print(social_data["social"].value_counts())
# sys.exit()

# text_data = social_data["text"].values.tolist()
# label = social_data[["social"]].values.tolist()

# X_train, X_test, y_train, y_test = train_test_split(text_data, label, test_size=0.2, random_state=42)

df_train, df_test = train_test_split(social_data, test_size=0.2, random_state=42)

# label_encoder = OneHotEncoder()
# y_train_embed = label_encoder.fit_transform(y_train)
# print(label_encoder.categories_, '\n')

# print(y_train_encode.shape)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=120):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label_encoder = OneHotEncoder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        label = self.df[['social']]
        label = self.label_encoder.fit_transform(label).toarray()[idx]

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label),
        }

# # Distribution of length of Sentence => max_lenght of this data is 255
# all_data = X_train + X_test
# encoded_text = [tokenizer.encode(text, add_special_tokens=True) for text in all_data]
# token_lens = [len(text) for text in encoded_text]
# sns.displot(token_lens)
# plt.xlim([0,max(token_lens)])
# plt.xlabel('Token Count')
# plt.show()

# Model
class TextClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

# Train & Eval Function
def train(model, criterion, optimizer, train_loader):
    model.train()
    losses = []
    correct = 0
    correct2 = 0
    check = 0

    for data in train_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        pred = pred.cpu()
        pred = pred.reshape(-1,1)

        row, col = pred.shape
        pred_encode = torch.zeros(row, 4)
        for i in range(row):
            for j in range(col):
                pred_encode[i,pred[i,j]] = 1

        pred_encode = pred_encode.to(device)

        count, _ = targets.shape
        for i in range(count):
            correct += torch.sum(torch.all(pred_encode[i] == targets[i]))

        targets = targets.cpu()
        pred_encode = pred_encode.cpu()
        row, col = targets.shape
        str_target = torch.zeros(row)
        str_pred = torch.zeros(row)
        for i in range(row):
            for j in range(col):
                if(targets[i,j] != 0):
                    str_target[i] = j
                if(pred_encode[i,j] != 0):
                    str_pred[i] = j

        correct2 += f1_score(str_target.numpy(), str_pred.numpy(), average='micro')
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        check+=1

    print(f'F1_Score: {correct2/check:.2f} Train Accuracy: {correct.double()/len(train_loader.dataset):.2f} Loss: {np.mean(losses):.2f}')
    return correct.double()/len(train_loader.dataset)

""" def eval(model, criterion, test_loader, valid_loader, test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)
            targets = data['targets'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            pred = pred.cpu()
            pred = pred.reshape(-1,1)

            row, col = pred.shape
            pred_encode = torch.zeros(row, 4)
            for i in range(row):
                for j in range(col):
                    if(j == pred[i,j].numpy()):
                        pred_encode[i,j] = 1

            # pred.to(device)
            pred_encode = pred_encode.to(device)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred_encode == targets)
            losses.append(loss.item())
    
    if test_data:
        print(f'Test Accuracy: {correct.double()/len(test_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(test_loader.dataset)
    else:
        print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(valid_loader.dataset) """

# Test Fuction
def test(data_loader):
    models = []
    model = TextClassifier(n_classes=4).to(device)
    model.load_state_dict(torch.load(f'model_social.pth'))
    models.append(model)

    texts = []
    predicts = []
    predict_probs = []
    real_values = []

    for data in data_loader:
        text = data['text']
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        total_outs = []
        for model in models:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                total_outs.append(outputs)
        
        total_outs = torch.stack(total_outs)
        _, pred = torch.max(total_outs.mean(0), dim=1)

        pred = pred.cpu()
        pred = pred.reshape(-1,1)

        row, col = pred.shape
        pred_encode = torch.zeros(row, 4)
        for i in range(row):
            for j in range(col):
                pred_encode[i,pred[i,j]] = 1

        pred_encode = pred_encode.to(device)

        # count, _ = targets.shape
        # for i in range(count):
        #     correct += torch.sum(torch.all(pred_encode[i] == targets[i]))
        
        texts.extend(text)
        predicts.extend(pred_encode)
        predict_probs.extend(total_outs.mean(0))
        real_values.extend(targets)
    
    predicts = torch.stack(predicts).cpu()
    predict_probs = torch.stack(predict_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    print(classification_report(real_values, predicts))
    return real_values, predicts

train_dataset = TextDataset(df_train, tokenizer, max_len=255)

test_dataset = TextDataset(df_test, tokenizer, max_len=255)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)


# Train Step
model = TextClassifier(n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)*n_epoch
            )

best_acc = 0
for epoch in range(n_epoch):
    print(f'Epoch {epoch+1}/{n_epoch}')

    temp_acc = train(model, criterion, optimizer, train_loader)   
    lr_scheduler.step()

    if temp_acc > best_acc:
        torch.save(model.state_dict(), f'model_social.pth')
        best_acc = temp_acc
    
    print('-'*30)


# Test Step
real_values, predicts = test(test_loader)