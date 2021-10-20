import numpy as np
import torch
from model import Model
import csv
from dataset import CowsDataset

# SETTINGS
NUM_EPOCHS = 10000

with open('cows.csv','r') as f:
    raw = csv.reader(f, delimiter=',')
    data = np.empty((len(list(raw))-1,8))

with open('cows.csv','r') as f:
    raw = csv.reader(f, delimiter=',')
    next(raw, None)
    for i, row in enumerate(raw):
        data[i] = np.array([float(el) for el in row], dtype=np.float32)


model = torch.load('model_3layers_crossvalidation.mdl')

with torch.no_grad():
    model.eval()
    with open('compare_crossvalidated.csv', 'w') as f:
        f.write('Predicted; Actual\n')
        for i in range(data.shape[0]):
            pred = model(torch.Tensor(data[i,:-1]).float().unsqueeze(0))
            actual = data[i,-1]
            f.write(f'{pred.item():.4f};{actual:.4f}'+'\n')
            # print(pred.item(), actual)
