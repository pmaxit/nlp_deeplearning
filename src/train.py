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
from pytorch_lightning.metrics.classification import AUROC
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import SpookyDataModule

N_EPOCHS = 10
BATCH_SIZE = 32
BERT_MODEL_NAME='bert-base-cased'
LABEL_COLUMNS = ['A','B','C']
class TextClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, steps_per_epoch = None, n_epochs= None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, input_ids, attention_mask , labels=None):
        output = self.bert(input_ids, attention_mask = attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        loss, outputs = self(input_ids , attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss,'predictions': outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        loss, outputs = self(input_ids , attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return {'val_loss': loss,'predictions': outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        loss, outputs = self(input_ids , attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return {'test_loss': loss,'predictions': outputs, "labels": labels}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []

        for output in outputs:
            labels.append(output['labels'].detach().cpu())
            predictions.append(output['predictions'].detach().cpu())

        labels = np.stack(labels)
        predictions = np.stack(predictions)

        #accuracy = (labels == predictions).mean()
        #self.logger.experiment.add_scaler("accuracy", accuracy, self.current_epoch)
        #roc_score = roc_auc_score(labels, predictions, multi_class='ovr',labels=LABEL_COLUMNS)
        #self.logger.experiment.add_scaler(f"{name}_roc_auc/Train",roc_score, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        # use roughy 1/3rd of example is for warm up and then this point onwards we will go back to zero
        warm_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warm_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warm_steps, total_steps)

        return [optimizer], [scheduler]

if __name__ == '__main__':
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    lbl_enc = preprocessing.LabelEncoder()
    train_df['enc_author'] = lbl_enc.fit_transform(train_df['author'].values)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    dm = SpookyDataModule(train_df, test_df,tokenizer)
    dm.setup()

    sample_item = dm.train_set[0]

    model = TextClassifier(n_classes=3, steps_per_epoch=len(train_df)//BATCH_SIZE, n_epochs= N_EPOCHS )

    _ , predictions = model(
        sample_item['input_ids'].unsqueeze(dim=0),
        sample_item['attention_mask'].unsqueeze(dim=0)
    )
    print("predictions " , predictions)

    trainer = pl.Trainer(max_epochs=N_EPOCHS, fast_dev_run=True, progress_bar_refresh_rate=20)
    trainer.fit(model, dm)

