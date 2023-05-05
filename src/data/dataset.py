import os
from typing import Dict

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

with open('../config/cfg1.yaml') as f:
    config = yaml.safe_load(f)


class ClipForImageSearchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str = 'train') -> None:
        super().__init__()
        self.df = df # [image, title]
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(config['BERT']['TOKENIZER_CHECKPOINT'])
        if self.split == 'train':
            self.transform = A.Compose([
                A.Resize(height=config['ViT']['TRANSFORMS']['HEIGHT'], width=config['ViT']['TRANSFORMS']['WIDTH']),
                ToTensorV2(),
            ])
        elif self.split == 'test':
            self.transform = A.Compose([
                A.Resize(height=config['ViT']['TRANSFORMS']['HEIGHT'], width=config['ViT']['TRANSFORMS']['WIDTH']),
                ToTensorV2(),
            ])
        else:
            raise NotImplemented(f"Split {self.split} has not been implemented as of yet, please try either of ['train', 'test']")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx, :]
        text = row['title']
        image_path = row['image']

        inputs = self.tokenizer(
            text=text, return_tensors='pt', 
            truncation=True, padding='max_length', 
            max_length=config['BERT']['MAX_LENGTH'], return_attention_mask=True, 
            return_token_type_ids=False
        )

        inputs = {k:v.squeeze() for k, v in inputs.items()}

        if self.split == 'train':
            image = np.array(Image.open(os.path.join(config['DATA_DIR'], 'train_images', image_path)).convert('RGB'))
        else:
            image = np.array(Image.open(os.path.join(config['DATA_DIR'], 'test_images', image_path)).convert('RGB'))

        image = self.transform(image=image)['image']
        inputs['image'] = image
        return inputs


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config['DATA_DIR'], 'train.csv'))
    dataset = ClipForImageSearchDataset(df=df, split='train')
    print(f'Length: {len(dataset)}')
    batch = dataset[0]
    __import__('pprint').pprint({k:v.shape for k,v in batch.items()})

    dataloader = DataLoader(dataset=dataset, batch_size=config['HYPERPARAMETERS']['TRAIN_BS'], shuffle=True)

    batch = next(iter(dataloader))

    __import__('pprint').pprint({k:v.shape for k,v in batch.items()})
