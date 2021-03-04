from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import pandas as pd
import torch
#from pytorch_lightning.metrics.functional.classification import aucroc
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Create data modules for the dataset
class SpookyAuthorDataset(Dataset):
    def __init__(self, data:pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128, label=True):
        self. data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row.text
        if self.label==False:
            label = -1.
        else:
            label = data_row.enc_author

        encoding = self.tokenizer.encode_plus(text, 
                            add_special_tokens= True,
                            max_length=self.max_token_len,
                            return_token_type_ids=False,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt')
        
        return dict(
             text= text,
             label= label,
             input_ids=encoding['input_ids'].flatten(),
             attention_mask=encoding['attention_mask'].flatten()
        )

class SpookyDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
    
    def setup(self):
        full_train = SpookyAuthorDataset(self.train_df, self.tokenizer, self.max_token_len)
        train_length = int(len(full_train)*0.8)

        self.train_set, self.val_set = torch.utils.data.random_split(full_train, [train_length, len(full_train) - train_length])
        self.test_dataset = SpookyAuthorDataset(self.test_df, self.tokenizer, self.max_token_len, label=False)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size, shuffle=True,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size = 1, shuffle=False, num_workers=4)

    
if __name__ == '__main__':
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    lbl_enc = preprocessing.LabelEncoder()
    
    train_df['enc_author'] = lbl_enc.fit_transform(train_df['author'].values)


    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    dm = SpookyDataModule(train_df, test_df,tokenizer)
    dm.setup()
    print(dm.train_set[0])
    print(next(iter(dm.train_dataloader())))

