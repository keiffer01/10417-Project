import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

def genre2Num(genre):
  classes = {
    "blues":     0,
    "classical": 1,
    "country":   2,
    "disco":     3,
    "hiphop":    4,
    "jazz":      5,
    "metal":     6,
    "pop":       7,
    "reggae":    8,
    "rock":      9
  }
  return classes.get(genre)

class ImageCsvDataset(Dataset):
  def __init__(self, csv_file, image_dir, transform=None):
    """
      Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
          on a sample.
    """
    self.image_dir = image_dir
    self.transform = transform

    features = pd.read_csv(csv_file)
    scaledFeatures = features.drop(["filename", "label", "length"], axis=1)
    scaledFeatures = ((scaledFeatures - scaledFeatures.min()) /
                      (scaledFeatures.max() - scaledFeatures.min()))
    scaledFeatures["filename"] = features["filename"]
    scaledFeatures["label"] = features["label"]
    self.music_features = scaledFeatures

  def __len__(self):
    return len(self.music_features)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    # Get image
    filename = self.music_features.loc[idx, "filename"]
    genre = filename.partition(".")[0]
    img_name = os.path.join(self.image_dir,
                            genre,
                            filename.replace(".", "", 1).replace(".wav", ".png"))
    image = Image.open(img_name)

    # Get corresponding features.
    music_features = self.music_features.drop(["filename", "label"], axis=1)
    features = music_features.loc[idx, :]
    features = np.array(features)
    features = features.astype("float")
    features = torch.tensor(features, dtype=torch.float32)

    # Image transformation
    if self.transform:
      image = self.transform(image)

    sample = {"image": image, "features": features}
    genre = torch.tensor(genre2Num(genre))
    return sample, genre
