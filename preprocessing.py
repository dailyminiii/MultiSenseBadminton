############
# MIT License
#
# Copyright (c) 2023 Minwoo Seong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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