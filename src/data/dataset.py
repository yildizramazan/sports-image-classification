# Custom PyTorch Dataset definition


from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os

class SportsDataset(Dataset):

    def __init__(self, csv_path, img_dir, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transforms = transforms
        self.classes = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_ID"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Critical Error: Image not found at '{img_path}'! Please check the file path and the CSV.")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Critical Error: File '{img_path}' could not be opened or is corrupted! Detail: {e}")


        label = self.class_to_idx[row["label"]]

        if self.transforms:
            image = self.transforms(image)
        
        return image,label
        

def get_class_distribution(dataset):
    return Counter(dataset.df["label"])



def get_sample_images(dataset, n_per_class=3):
    sample_images = {}
    for cls in dataset.classes:
        sampled_rows = dataset.df[dataset.df["label"] == cls].sample(n_per_class)
        imgs = []
        for index, row in sampled_rows.iterrows():
            img_path = os.path.join(dataset.img_dir, row["image_ID"])
            image = Image.open(img_path).convert("RGB")
            imgs.append(image)
        sample_images[cls] = imgs
    return sample_images