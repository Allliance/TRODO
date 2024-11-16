import os, random
from torch.utils.data import Dataset
from copy import copy

CLEAN_LABEL = 0

class ModelDataset(Dataset):
    def __init__(self,
                 clean_folder,
                 trojaned_folder,
                 model_loader,
                 sample=False,
                 sample_k=5,
                 ):
        
        self.models_data = []
        for model_file in os.listdir(clean_folder):
            if model_file.lower().endswith(".pt"):
                self.models_data.append({
                    'path': os.path.join(clean_folder, model_file),
                    'label': CLEAN_LABEL,
                })
                
            
        for model_file in os.listdir(trojaned_folder):
            if model_file.lower().endswith(".pt"):
                self.models_data.append({
                    'path': os.path.join(trojaned_folder, model_file),
                    'label': 1 - CLEAN_LABEL,
                })

    def load_model(self, model_data):
        return self.loader(model_data['path'], model_data)

    def get_random_clean_model(self):
        return self.load_model(random.sample(self.cleans_data, 1)[0])
    
    def get_random_bad_model(self, name=None):
        if name is None:
            model_data = random.sample(self.bads_data, 1)[0]
        else:
            model_data = random.sample(self.model_data_dict[name], 1)[0]
        return self.load_model(model_data)

    def __len__(self):
        return len(self.models_data)

    def __getitem__(self, idx):
        model_data = self.models_data[idx]
        model = self.loader(model_data['path'], model_data)
        label = model_data['label']
        
        return model, label
    