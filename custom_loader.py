import torch 
import h5py

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        #load dataset keys
        tmp = h5py.File(path, 'r')
        group_key1 = list(tmp.keys())[0]
        group_key2 = list(tmp.keys())[1]

        #save data and labels
        self.data = tmp[group_key1]
        self.labels = tmp[group_key2]

        #transform to tensor flag
        self.transform = transform

        
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)
            labels = self.transform(labels)
        
        return data, labels
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.labels)
    