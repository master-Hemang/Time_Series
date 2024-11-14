import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, path, data_type='train'):
        # Load the data from the specified path
        self.data = torch.load(path)
        
        # Check if data is a dictionary; if so, look for the required key
        if isinstance(self.data, dict):
            if data_type in self.data:
                self.data = self.data[data_type]
            else:
                raise KeyError(f"'{data_type}' key not found in the data dictionary.")
        # If not a dictionary, assume it’s directly the data we need
        elif isinstance(self.data, torch.Tensor) or isinstance(self.data, np.ndarray):
            print(f"Loaded data directly as a {type(self.data).__name__}.")
        else:
            raise ValueError("Data format not recognized. Expected a dictionary, tensor, or NumPy array.")

    def preprocess(self, scale=True):
        # Ensure the data is in numpy format for processing
        if isinstance(self.data, torch.Tensor):
            data_np = self.data.numpy()  # Convert tensor to numpy array for preprocessing
        elif isinstance(self.data, np.ndarray):
            data_np = self.data
        else:
            raise ValueError("Data should be a tensor or NumPy array for preprocessing")

        # Handle missing values by replacing NaNs with 0
        data_np = np.nan_to_num(data_np)

        # Scale the data if the scale flag is True
        if scale:
            scaler = StandardScaler()
            shape = data_np.shape
            data_np = data_np.reshape(-1, shape[-1])  # Flatten to 2D for scaling
            data_np = scaler.fit_transform(data_np)  # Fit and transform the data
            data_np = data_np.reshape(shape)  # Reshape back to original shape

        # Convert the processed data back to a tensor
        self.data = torch.tensor(data_np)  # Update self.data with preprocessed tensor
        return self.data

    def get_data(self):
        return self.data  # Return the processed data for use in model training

# Example usage:
# loader = DataLoader('Gesture/train.pt', data_type='train')
# preprocessed_data = loader.preprocess(scale=True)
# print("Processed data:", preprocessed_data)
