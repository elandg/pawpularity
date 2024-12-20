import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm


class PawpularityDataset(Dataset):
    """Pawpularity dataset."""

    def __init__(self, cf, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing Pawpularity information
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df_raw = pd.read_csv(cf["data"]["csv_path"])
        num_bins = cf["data"].get("num_bins", 100)

        # Change labels to 0-99
        df_binned = df_raw.assign(Pawpularity=df_raw["Pawpularity"] - 1 //
                                    (100 // num_bins))

        self.df = df_binned
        self.img_dir = cf["data"]["img_dir"]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        metadata = self.df.iloc[idx, 1:13].values.astype(np.uint8)

        img_path = os.path.join(self.img_dir, self.df.at[idx, "Id"] + ".jpg")
        image = read_image(img_path).to(torch.float32) / 255
        if self.transform:
            image = self.transform(image)

        # score = torch.tensor(self.df.at[idx, 'Pawpularity'] - 1).float()
        score = self.df.at[idx, "Pawpularity"]

        return {"image": image, "metadata": metadata, "score": score}

    def get_img_sizes(self, fresh_run=True, append_to_df=True):
        if fresh_run:
            self.df = self.df.drop(columns=["Shape"], errors="ignore")

        img_sizes = []
        if "Shape" not in self.df.columns:
            for i in tqdm(range(len(self))):
                img = self[i]["image"]
                img_sizes.append(list(img.shape))
            if append_to_df:
                self.df.loc[:, "Shape"] = img_sizes
        else:
            img_sizes = self.df["Shape"]
        return torch.tensor(img_sizes)


def bin_csv(csv_path_callable: callable,
            num_bins: int = 100,
            save_binned_csv: bool = False) -> pd.DataFrame:
    """
    Loads and bins base csv, which has labels of Pawpularity scores 1-100
    Avoid binning by passing num_bins = 100 or leaving it default
    """
    if num_bins and not (1 <= num_bins <= 100):
        logging.warning(
            f"num_bins should be between 1 and 100. Pawpularity ranges from 1-100 and is rescaled to 0-99"
        )

    df_raw = pd.read_csv(csv_path_callable("train.csv"))

    # Change labels to 0-99
    df_binned = df_raw.assign(Pawpularity=df_raw["Pawpularity"] - 1 //
                              (100 // num_bins))

    if save_binned_csv:
        df_binned.to_csv(csv_path_callable(f"train_{num_bins}_bins.csv"),
                         index=False)

    return df_binned


def main():
    pass


#     with open(os.path.join(ROOT, "config/config.yaml")) as f:
#         cf = yaml.safe_load(f)

#     num_bins = cf["data"]["num_bins"]
#     data_path_func = partial(os.path.join, DATA_PATH)
#     df = bin_csv(data_path_func)
#     # Parameters for "Data Analysis"
#     images = data_path_func("images")
#     img_size = [cf["data"]["augs"]["resize"]] * 2
#     dset = PawpularityDataset(df, images)
#     toy_dset = PawpularityDataset(df[:100], images)

if __name__ == "__main__":
    main()
