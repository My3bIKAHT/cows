import numpy as np
import torch
from model import Model
import csv
from dataset import CowsDataset
from torch.utils.tensorboard import SummaryWriter

# SETTINGS
NUM_EPOCHS = 1000

with open('cows.csv','r') as f:
    raw = csv.reader(f, delimiter=',')
    data = np.empty((len(list(raw))-1,8))

with open('cows.csv','r') as f:
    raw = csv.reader(f, delimiter=',')
    next(raw, None)
    for i, row in enumerate(raw):
        data[i] = np.array([float(el) for el in row], dtype=np.float32)


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
writer = SummaryWriter()

np.random.shuffle(data)
ctdata = data[:-100:,:]
cvdata = data[-100:,:]


for epoch in range(NUM_EPOCHS):
    print(f'Epoch: {epoch}')
    t_losses, v_losses = [], []

    params = {
                'batch_size': 32,
                'shuffle': True,
                'num_workers': 0
            }
    trainingSet = CowsDataset(ctdata)
    trainingGenerator = torch.utils.data.DataLoader(trainingSet, **params)

    for b_data, b_yield in trainingGenerator:
        optimizer.zero_grad()
        pred = model(b_data)
        loss = criterion(pred.flatten(), b_yield)
        loss.backward()
        optimizer.step()
        t_losses.append(loss.item())

    validationSet = CowsDataset(cvdata)
    validationGenerator = torch.utils.data.DataLoader(validationSet, **params)
    with torch.no_grad():
        model.eval()
        for v_data, v_yield in validationGenerator:
            pred = model(v_data)
            loss = criterion(pred.flatten(), v_yield)
            v_losses.append(loss.item())
        model.train()

    print(f"Loss, training: {np.mean(t_losses):.4f}, validation: {np.mean(v_losses):.4f}")
    writer.add_scalar('Loss/train', np.mean(t_losses), epoch)
    writer.add_scalar('Loss/valid', np.mean(v_losses), epoch)

torch.save(model, 'model_3layers.mdl')