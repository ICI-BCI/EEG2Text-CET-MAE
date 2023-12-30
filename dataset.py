from torch.utils.data import Dataset
import os
import pickle

# 如果要改input_embeddings变为 normalized_input_embeddings 在这里
class EEG_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), 'rb') as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample['input_embeddings'],
            input_sample['non_normalized_input_embeddings'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            file
        )

class EEG_dataset_add_sentence(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), 'rb') as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample['input_embeddings'],
            input_sample['normalized_input_embeddings'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            file
        )


class EEG_dataset_add_sentence_clip(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), 'rb') as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample['input_embeddings'],
            input_sample['normalized_input_embeddings'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['mvp_target_tokenized'],
            file
        )

class EEG_dataset_add_sentence_mae(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.path, file), 'rb') as handle:
            input_sample = pickle.load(handle)
        return (
            input_sample['input_embeddings'],
            input_sample['normalized_input_embeddings'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['mvp_target_tokenized'],
            input_sample['target_string']
        )