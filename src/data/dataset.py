import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ReverseImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df # [image_path, text]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx, :]
        text = row['text']
        image_path = row['image_path']

        return {
            'text': text, 
            'image': image_path
        }