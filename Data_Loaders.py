import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
       # Separate data by label
        label_0_data = self.data[self.data[:, -1] == 0]  # Samples with label 0
        label_1_data = self.data[self.data[:, -1] == 1]  # Samples with label 1

        # Print initial counts of each label
        print(f"Initial count of label 0: {len(label_0_data)}")
        print(f"Initial count of label 1: {len(label_1_data)}")

        # Identify majority and minority class
        if len(label_0_data) > len(label_1_data):
            majority_class_data = label_0_data
            minority_class_data = label_1_data
        else:
            majority_class_data = label_1_data
            minority_class_data = label_0_data

        # Check if the majority class is more than 4 times the minority class
        if len(majority_class_data) > 4 * len(minority_class_data):
            target_majority_count = 4 * len(minority_class_data)
            # Downsample the majority class to 4 times the minority class size
            majority_class_data = majority_class_data[np.random.choice(len(majority_class_data), target_majority_count, replace=False)]
            print(f"Downsampling majority class to {target_majority_count} samples.")
        else:
            print("No downsampling needed. Majority class is within 4:1 ratio.")

        # Combine the adjusted datasets
        balanced_data = np.vstack((majority_class_data, minority_class_data))

        # Shuffle the data
        np.random.shuffle(balanced_data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        x = torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.normalized_data[idx, -1], dtype=torch.float32)
        
        return {'input': x, 'label': y}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        # Split dataset (80% train, 20% test)
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size
        train_dataset, test_dataset = data.random_split(self.nav_dataset, [train_size, test_size])

        # Create data loaders
        self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
