import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataLoader:
    def __init__(self, path):
        # Load the data from the given path
        self.data = torch.load(path)

        # Inspect the structure of the data
        print("Loaded data structure:", type(self.data))
        if isinstance(self.data, dict):  # If it's a dictionary, print the keys
            print("Keys in data:", self.data.keys())
            print("Sample of data:", self.data)
            # Access the correct data subset (adjust the key as per your data structure)
            self.data = self.data.get('train_data', None)  # Adjust key name as needed
            if self.data is None:
                print("Error: 'train_data' not found in the loaded data!")
        else:
            print("Data is not a dictionary, it is:", type(self.data))
            # Handle cases where data is not a dictionary
            self.data = self.data  # If it's a tensor or other structure, keep it as it is

    def preprocess(self, scale=True):
        if isinstance(self.data, torch.Tensor):
            data_np = self.data.numpy()  # Convert tensor to numpy array
        else:
            raise ValueError("Data should be a tensor after preprocessing")
        
        # Handle missing values by replacing NaNs with 0
        data_np = np.nan_to_num(data_np)

        # Optionally scale the data (standardization)
        if scale:
            scaler = StandardScaler()
            shape = data_np.shape
            data_np = data_np.reshape(-1, shape[-1])  # Flatten to 2D for scaling
            data_np = scaler.fit_transform(data_np)  # Fit and transform the data
            data_np = data_np.reshape(shape)  # Reshape back to original shape

        return torch.tensor(data_np)  # Convert back to tensor


# Usage example for gesture data
gesture_train = DataLoader('Gesture/train.pt').preprocess()
gesture_test = DataLoader('Gesture/test.pt').preprocess()
gesture_val = DataLoader('Gesture/val.pt').preprocess()

# For HAR data
har_train = DataLoader('HAR/train.pt').preprocess()
har_test = DataLoader('HAR/test.pt').preprocess()
har_val = DataLoader('HAR/val.pt').preprocess()
