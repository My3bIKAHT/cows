from torch.utils.data import Dataset
import numpy as np

class CowsDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):       
        return self.data[index, :-1].astype(np.float32), self.data[index,-1].astype(np.float32)