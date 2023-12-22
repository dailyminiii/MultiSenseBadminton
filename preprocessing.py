import torch
from torch.utils.data import Dataset

# Define a custom dataset class
class BadmintonDataset(Dataset):
    def __init__(self, feature_matrices,
                        feature_matrices_ground_truth
                ):
        
        self.data = feature_matrices
        self.ground_truth = feature_matrices_ground_truth


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.ground_truth[index], dtype=torch.float32)
            
        return x, y