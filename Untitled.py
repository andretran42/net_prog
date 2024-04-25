#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import os
import string
import matplotlib.pyplot as plt
from os import listdir
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

# from keras.optimizers import Adam
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint

import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from tqdm.notebook import tqdm

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from multiprocessing import cpu_count


# In[2]:


directory = './data/'
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
frames = {}
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    df_name = os.path.splitext(csv_file)[0]  # Use file name as DataFrame name (without extension)
    df = pd.read_csv(file_path)
    frames[df_name] = df


# In[3]:


for title, frame in frames.items():
    translation_table = str.maketrans('', '', string.digits)
    sitename = title.split("_")[0].translate(translation_table)
    frame['Site'] = sitename


# In[4]:


def remove_numbers(input_string):
    if type(input_string) == float:
        return input_string
    if 'Application Data' in input_string:
        return 'Application Data'
    if 'Payload' in input_string:
        return 'Payload'
    if 'Handshake' in input_string:
        return 'Handshake'
    translation_table = str.maketrans('', '', '0123456789')
    return input_string.translate(translation_table)


# In[5]:


def info_dic_map(input_string):
    return info_dic[input_string]


# In[6]:


def apply_target(df):
    if df['Site'].iloc[0] == ('amazon' or 'chatgpt' or 'bing' or 'discord' or 'googledrive' or 'wmregistration' or 'quizlet'):
        df['Target'] = 'sud'
        return df
    elif df['Site'].iloc[0] == ('discordstream' or 'fortnite' or 'minecraft'):
        df['Target'] = 'cud'
        return df
    elif df['Site'].iloc[0] == ('youtube' or 'hulu'):
        df['Target'] = 'cd'
        return df
    else:
        df['Target'] = 'sd'
        return df


# In[7]:


def extract_values(text):
    if type(text) == (float or Int):
        return text
    # Use regex pattern to find values enclosed within square brackets
    pattern = r'\[([^\]]+)\]'  # Matches anything inside square brackets
    matches = re.findall(pattern, text)
    if not matches:
        return text
    return matches[0]


# In[8]:


site_dic = {}
info_dic = {}
j = 0
for title, frame in frames.items():
    frame['Time Delta'] = frame['Time'].diff().fillna(0)
    #1 = user
    #0 = site
    
    frame['Source'] = frame['Source'].str.replace('18.160.17.214', '0')
    frame['Destination'] = frame['Destination'].str.replace('18.160.17.214', '0')
    frame['Source'] = frame['Source'].str.replace('172.66.40.147', '0')
    frame['Destination'] = frame['Destination'].str.replace('172.66.40.147', '0')
    frame['Source'] = frame['Source'].str.replace('104.234.169.167', '0')
    frame['Destination'] = frame['Destination'].str.replace('104.234.169.167', '0')
    frame['Source'] = frame['Source'].str.replace('162.159.128.232', '0')
    frame['Destination'] = frame['Destination'].str.replace('162.159.128.232', '0')
    frame['Source'] = frame['Source'].str.replace('100.69.171.196', '1')
    frame['Destination'] = frame['Destination'].str.replace('100.69.171.196', '1')
    frame['Source'] = frame['Source'].str.replace('169.254.243.145', '0')
    frame['Destination'] = frame['Destination'].str.replace('169.254.243.145', '0')
    frame['Source'] = frame['Source'].str.replace('192.168.0.92', '1')
    frame['Destination'] = frame['Destination'].str.replace('192.168.0.92', '1')
    frame['Source'] = frame['Source'].str.replace('192.168.0.129', '1')
    frame['Destination'] = frame['Destination'].str.replace('192.168.0.129', '1')
    '169.254.243.145'
    

    #amazon
    frame['Source'] = frame['Source'].str.replace('204.79.197.200', '0')
    frame['Destination'] = frame['Destination'].str.replace('204.79.197.200', '0')

    #chatgpt
    frame['Source'] = frame['Source'].str.replace('104.18.37.228', '0')
    frame['Destination'] = frame['Destination'].str.replace('104.18.37.228', '0')

    #discord / discordstream
    frame['Source'] = frame['Source'].str.replace('66.22.231.191', '0')
    frame['Destination'] = frame['Destination'].str.replace('66.22.231.191', '0')
    frame['Source'] = frame['Source'].str.replace('66.22.196.159', '0')
    frame['Destination'] = frame['Destination'].str.replace('66.22.196.159', '0')
    frame['Source'] = frame['Source'].str.replace('35.214.213.22', '0')
    frame['Destination'] = frame['Destination'].str.replace('35.214.213.22', '0')

    frame['Source'] = frame['Source'].str.replace('23.48.104.108', '0')
    frame['Destination'] = frame['Destination'].str.replace('23.48.104.108', '0')

    

    #wmreg
    frame['Source'] = frame['Source'].str.replace('3.14.34.81', '0')
    frame['Destination'] = frame['Destination'].str.replace('3.14.34.81', '0')
    frame['Source'] = frame['Source'].str.replace('100.86.171.196', '1')
    frame['Destination'] = frame['Destination'].str.replace('100.86.171.196', '1')

    #fortnite
    frame['Source'] = frame['Source'].str.replace('3.144.65.185', '0')
    frame['Destination'] = frame['Destination'].str.replace('3.144.65.185', '0')

    #googledrive
    frame['Source'] = frame['Source'].str.replace('172.253.122.139', '0')
    frame['Destination'] = frame['Destination'].str.replace('172.253.122.139', '0')

    #minecraft
    frame['Source'] = frame['Source'].str.replace('209.222.115.47', '0')
    frame['Destination'] = frame['Destination'].str.replace('209.222.115.47', '0')

    #quizlet
    frame['Source'] = frame['Source'].str.replace('104.16.133.27', '0')
    frame['Destination'] = frame['Destination'].str.replace('104.16.133.27', '0')

    #hulu
    frame['Source'] = frame['Source'].str.replace('23.48.104.112', '0')
    frame['Destination'] = frame['Destination'].str.replace('23.48.104.112', '0')
    frame['Source'] = frame['Source'].str.replace('192.168.0.104', '1')
    frame['Destination'] = frame['Destination'].str.replace('192.168.0.104', '1')
    
    frame['Source'] = frame['Source'].str.replace('20.36.181.22', '0')
    frame['Destination'] = frame['Destination'].str.replace('20.36.181.22', '0')
    frame['Source'] = frame['Source'].str.replace('192.168.0.200', '1')
    frame['Destination'] = frame['Destination'].str.replace('192.168.0.200', '1')

    #ryrod / ugm / wpbegin
    frame['Source'] = frame['Source'].str.replace('100.86.7.137', '1')
    frame['Destination'] = frame['Destination'].str.replace('100.86.7.137', '1')
    frame['Source'] = frame['Source'].str.replace('104.18.10.41', '0')
    frame['Destination'] = frame['Destination'].str.replace('104.18.10.41', '0')
    frame['Source'] = frame['Source'].str.replace('172.66.43.109', '0')
    frame['Destination'] = frame['Destination'].str.replace('172.66.43.109', '0')
    frame['Source'] = frame['Source'].str.replace('13.84.36.2', '0')
    frame['Destination'] = frame['Destination'].str.replace('13.84.36.2', '0')
    frame['Source'] = frame['Source'].str.replace('208.80.154.224', '0')
    frame['Destination'] = frame['Destination'].str.replace('208.80.154.224', '0')

    #youtube
    frame['Source'] = frame['Source'].str.replace('2600:8805:3e22:4100:68dd:8e31:1a2:1f62', '1')
    frame['Destination'] = frame['Destination'].str.replace('2600:8805:3e22:4100:68dd:8e31:1a2:1f62', '1')
    frame['Source'] = frame['Source'].str.replace('2607:f8b0:4004:f::a', '0')
    frame['Destination'] = frame['Destination'].str.replace('2607:f8b0:4004:f::a', '0')

    frame['Info'] = frame['Info'].apply(extract_values)

    for i in frame['Destination'].unique():
        if (i != '1' and i!='0'):
            print(i)
            frame['Source'] = frame['Source'].str.replace(i, '0')
            frame['Destination'] = frame['Destination'].str.replace(i, '0')

    frame = apply_target(frame)

    protocol_map = {'UDP':0, 'TCP':1, 'TLSv1.2':2, 'TLSv1.3':3, 'RTCP': 4, 'QUIC': 5, 'SSDP':6, 'R-GOOSE':7}
    frame['Protocol'] = frame['Protocol'].replace(protocol_map)
    
    translation_table = str.maketrans('', '', string.digits)
    sitename = title.split("_")[0].translate(translation_table)
    frame['Info'] = frame['Info'].apply(remove_numbers)
    for value in frame['Info'].unique():
        if value not in info_dic:
            j+=1
            info_dic[value] = j

    frame['Info'] = frame['Info'].apply(info_dic_map)
    
    if sitename not in site_dic:
        site_dic[sitename] = 0
    rows_per = 50
    total_dataframes = (len(frame)) // rows_per
    for i in range(total_dataframes):
        site_dic[sitename] += 1
        file_path = './data2/' + sitename + '_' + str("{:03d}".format(site_dic[sitename]) + '.csv')
        start_index = i * rows_per
        end_index = (i + 1) * rows_per
        smaller_df = frame.iloc[start_index:end_index]
        smaller_df.to_csv(file_path, index=False)


# In[ ]:





# In[9]:


print(info_dic)


# In[10]:


# UDP = 0
# TCP = 1
# TLSv1.2 = 2


# In[11]:


rows_per = 50
total_dataframes = (len(df) + rows_per - 1) // rows_per
smaller_dfs = []
for i in range(total_dataframes):
    start_index = i * rows_per
    end_index = (i + 1) * rows_per
    smaller_df = df.iloc[start_index:end_index]
    smaller_dfs.append(smaller_df)
    print(end_index)


# In[12]:


train_df = {'amazon':28, 'bing':110, 'chatgpt':26, 'discord':100, 'googledrive':20, 'discordstream':20, 'fortnite':30, 'minecraft': 63, 'ryrod':75, 'ugm': 42, 'wpbeginner': 20, 'Wikipedia':17, 'youtube': 38, 'hulu':45, 'quizlet':30, 'wmregistration':5}


# In[13]:


target_map = {"sd":0, "sud":1, "cd":2, "cud":3}


# In[14]:


# Path to the folder containing CSV files
folder_path = './data2/'

# List to store all dataframes from CSV files
all_dataframes = []
test_dfs = []

# Iterate over each file in the folder
i = 0
y_train = pd.DataFrame(columns=['Series', 'Target'])
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file into a dataframe
        df = pd.read_csv(file_path)
        df['Sample'] = i
        y_train = y_train._append({'Series': i, 'Target': target_map[df.iloc[0]['Target']]}, ignore_index=True)
        i+=1
        # Append the dataframe to the list
        all_dataframes.append(df)
            
# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Path to the new combined CSV file
output_csv_path = './data3/x_train.csv'

# Write the combined dataframe to a new CSV file
combined_df.to_csv(output_csv_path, index=False)
y_train.to_csv('./data3/y_train.csv', index=False)

print(f"Combined CSV file saved to: {output_csv_path}")


# In[15]:


class LSTMModel (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print("FORWARD")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 6  # Number of features in input sequence
hidden_size = 64  # Number of features in LSTM hidden state
num_layers = 2  # Number of LSTM layers
output_size = 1  # Number of output units (for regression tasks)

model5 = LSTMModel(input_size, hidden_size, num_layers, output_size)

print(model5)


# In[16]:


loss_func = nn.NLLLoss()
learning_rate = 0.001
optimizer = optim.Adam(model5.parameters(), lr=learning_rate)


# In[17]:


from torch.utils.data import DataLoader, TensorDataset


# In[18]:


# data2_path = './data2/'
# data2_files = [file for file in os.listdir(data2_path) if file.endswith('.csv')]
# datasets = []
# for file in data2_files:
#     file_path = os.path.join(data2_path, file)
#     df = pd.read_csv(file_path)
    
#     # Example: Assuming 'features' and 'target' are columns in the CSV
#     features = df.drop('Time', axis=1)  # Extract features (input data)
#     features = features.drop('Site', axis=1)  # Extract features (input data)
#     features = features.drop('No.', axis=1)  # Extract features (input data)
#     features = np.array(features.iloc[:,:-1].values)
#     target = df['Target'].values  # Extract target values
#     print(target)
    
#     # Convert to PyTorch tensors
#     features_tensor_list = [torch.tensor(arr, dtype=torch.float32) for arr in features]
#     target_tensor = torch.tensor(target, dtype=torch.float32)
    
#     # Create a TensorDataset for the CSV data
#     dataset = TensorDataset(features_tensor, target_tensor)
#     datasets.append(dataset)


# In[19]:


x_train = pd.read_csv('./data3/x_train.csv')
y_train = pd.read_csv('./data3/y_train.csv')
x_train['Source'] = x_train['Source'].astype(int)
x_train['Destination'] = x_train['Destination'].astype(int)


# In[20]:


x_train['Site'].unique()
print(x_train.info())


# In[21]:


FEATURE_COLUMNS = ['Source', 'Destination', 'Protocol', 'Length', 'Info', 'Time Delta']
FEATURE_COLUMNS


# In[22]:


sequences = []

for sample, group in x_train.groupby('Sample'):
    sequence_features = group[FEATURE_COLUMNS]
    
    label = y_train[y_train.Series == sample].iloc[0].Target
    print(label)
    
    sequences.append((sequence_features, label))


# In[23]:


print(sequences[1])


# In[24]:


train_sequences, test_sequences = train_test_split(sequences,test_size=0.2)


# In[25]:


len(train_sequences), len(test_sequences)


# In[26]:


for seq, label in test_sequences:
    print(torch.Tensor(seq.to_numpy()).shape)


# In[ ]:


class SurfaceDataset(Dataset):

  def __init__(self,sequences):
      self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self,idx):
    sequence, label = self.sequences[idx]
    return dict(
        sequence = torch.Tensor(sequence.to_numpy()),
        labels = torch.as_tensor(label)
    )

class SurfaceDataModule(pl.LightningDataModule):

  def __init__(self, train_sequences, test_sequences, batch_size=8):
    super().__init__()
    self.batch_size = batch_size
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences

  def setup(self, stage=None):
    self.train_dataset = SurfaceDataset(self.train_sequences)
    self.test_dataset = SurfaceDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=1
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=1
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=1
    )



class SequenceModel(nn.Module):
  def __init__(self,n_features, n_classes, n_hidden=256, n_layers=3):
    super().__init__()
    self.lstm = LSTMModel(input_size = n_features, hidden_size=n_hidden, num_layers = n_layers, output_size = 4)
    #   nn.LSTM(
    #     input_size=n_features,
    #     hidden_size=n_hidden,
    #     num_layers=n_layers,
    #     batch_first=True,
    #     dropout=0.75
    # )

    self.classifier = nn.Linear(n_hidden,n_classes)

  def forward(self,x):
    # self.lstm.flatten_parameters()
    _,(hidden,_) = self.lstm(x)

    out = hidden[-1]
    return self.classifier(out)

class SurfacePredictor(pl.LightningModule):

  def __init__(self,n_features:int, n_classes: int):
    super().__init__()
    self.model = SequenceModel(n_features, n_classes)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, x, labels=None):
    print(x.shape)
    output = self.model()
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["labels"]
    loss, outputs = self(sequences, labels)
    predictions = torch.argmax(outputs,dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}
    

  def validation_step(self, batch, batch_idx):
    print(batch)
    sequences = batch["sequence"]
    labels = batch["labels"]
    loss, outputs = self(sequences, labels)
      
    predictions = torch.argmax(outputs,dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def test_step(self, batch, batch_idx):
    
    sequences = batch["sequence"]
    labels = batch["labels"]
    loss, outputs = self(sequences, labels)
    predictions = torch.argmax(outputs,dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.0001)
N_EPOCHS = 250
BATCH_SIZE = 16

data_module = SurfaceDataModule(
  train_sequences,
  test_sequences,
  batch_size=BATCH_SIZE
)
model = SurfacePredictor(n_features=6,n_classes=4)
print(model)
trainer = pl.Trainer()
trainer.fit(model, data_module)


# In[ ]:


model2 = SequenceModel(n_features=6, n_classes=4)
x = torch.randn(16, 50, 6)
print("FORWARD")
output=model2()


# In[ ]:


torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

