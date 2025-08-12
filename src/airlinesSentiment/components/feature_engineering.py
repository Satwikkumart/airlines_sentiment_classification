from airlinesSentiment import logger
import zipfile
import os
from airlinesSentiment.utils.common import get_size
import gdown
import spacy
import string
import pandas as pd
from pathlib import Path
from airlinesSentiment.entity.config_entity import DataPreprocessingConfig
import pandas as pd
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx       ])
        return item
    
class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        self.mapping_labels = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.data = self._load_data()


        #set the default file output path
        self.output_file_path = Path(self.config.training_cleansed_data) / 'cleaned_tweets.csv'
        self.datasets_dir = Path(self.config.datasets_dir)

    def _load_data(self) -> pd.DataFrame:

        #check if file exists
        if not Path(self.config.training_data_file).exists():
            raise FileNotFoundError(f"File not found: {self.config.training_data_file}")
        

        #load the dataset
        data = pd.read_csv(self.config.training_data_file)
        logger.info(f"Dataset loaded from {self.config.training_data_file}")
        return data
    
    def text_process(self) -> None:

        self.data['cleaned_text'] = self.data['text'].apply(self._process_single_text)
        logger.info("Text processing completed")

    
    def _process_single_text(self, text: str) -> str:

        text = text.lower()

        text = text.translate(str.maketrans('', '', string.punctuation))

        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in self.stop_words]

        return ' '.join(tokens)

    def mapping_labels_func(self) -> None:
        self.data['label'] = self.data['airline_sentiment'].map(self.mapping_labels)
        logger.info('labels mapped to numerical values')

    def tokenize_text(self) -> None:
        self.data['tokenized'] = self.data['cleaned_text'].apply(
            lambda x: self.tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        )

        logger.info(f"Text tokenization completed")
    
    def save_data(self):
        self.data.to_csv(self.output_file_path, index=False)
        logger.info(f"Preprocessed data saved to {self.output_file_path}")



    def train_val_test_split(self, test_size: float = 0.3, val_size: float = 0.5, random_state: int = 42)  -> dict:
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            self.data['cleaned_text'].to_list(), self.data['label'].to_list(), test_size=test_size, random_state=random_state
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=val_size, random_state=random_state
        )
        logger.info("Data split into, train, test and validation")

        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
    
    def convert_to_tokenized_datasets(self, splits: dict) -> dict:
        train_encodings = self.tokenizer(splits['train']['texts'], truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(splits['val']['texts'], truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(splits['test']['texts'], truncation=True, padding=True, max_length=128)


        train_dataset = SentimentDataset(train_encodings, splits['train']['labels'])
        val_dataset = SentimentDataset(val_encodings, splits['val']['labels'])
        test_dataset = SentimentDataset(test_encodings, splits['test']['labels'])
        
        logger.info("PyTorch datasets created sucessfully")
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    
    def save_datasets(self, train_dataset, val_dataset, test_dataset):
        datasets_dir = Path(self.config.datasets_dir)
        datasets_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_dataset, datasets_dir / "train_dataset.pt")
        torch.save(val_dataset, datasets_dir / "val_dataset.pt")
        torch.save(test_dataset, datasets_dir / "test_dataset.pt")

        logger.info(f"Datasets saved to {datasets_dir}")