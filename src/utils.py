from time import time
import os
import torch
from torch.utils.data import DataLoader
import torchmetrics

from utils.dataset import PawpularityDataset

metrics = {
    "mse": torchmetrics.MeanSquaredError(),
    "rmse": torchmetrics.MeanSquaredError(squared=False)
}


def find_opt_num_workers(dset, batch_size):
    # Determine max number of CPU cores
    max_cores = os.cpu_count()
    opt_num_workers = 0
    min_time = float('inf')
    for num_workers in range(0, max_cores + 2, 2):
        start_time = time()
        dataloader = DataLoader(dset,
                                batch_size=batch_size,
                                num_workers=num_workers)

        for batch in dataloader:
            # Simulate training process (CPU or GPU)
            if torch.cuda.is_available():
                batch = batch["image"].cuda()  # Move to GPU if available
            # Simulate a simple operation
            _ = batch * 2

        end_time = time()
        timespan = end_time - start_time
        print(f"num_workers={num_workers}, time={timespan:.2f}s")
        if timespan < min_time:
            opt_num_workers = num_workers
            min_time = timespan
    return opt_num_workers


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch in dataloader:
        data = batch["image"]
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


# ImageNet
mean, std = torch.tensor([0.485, 0.456,
                          0.406]), torch.tensor([0.229, 0.224, 0.225])


def main():
    pass
    # transform = transforms.Compose([transforms.Resize(img_size)])
    # dset = PawpularityDataset(df, images, transform=transform)

    # # opt_num_workers = find_opt_num_workers(dset, cf["loader"]["bs"])
    # opt_num_workers = 8
    # print(f'Optimum num workers: {opt_num_workers}')

    # dataloader = torch.utils.data.DataLoader(dset,
    #                                          batch_size=cf["loader"]["bs"],
    #                                          shuffle=True,
    #                                          num_workers=opt_num_workers)
    # # mean, std = get_mean_and_std(dataloader)
    # # mean, std = torch.tensor([0.5188, 0.4840, 0.4462]), torch.tensor([0.2652, 0.2608, 0.2629])
    # mean, std = torch.tensor([0.485, 0.456,
    #                           0.406]), torch.tensor([0.229, 0.224,
    #                                                  0.225])  # ImageNet
    # print(f'mean: {mean}, std: {std}')


if __name__ == "__main__":
    main()
