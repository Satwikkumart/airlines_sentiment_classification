import spacy
import string
from nltk.corpus import stopwords
from typing import List
from airlinesSentiment.entity.config_entity import DataPreprocessingConfig

class DataPreprocessor:
    def __init__(self, config:DataPreprocessingConfig):
        self.config = config
        self.nlp = spacy.load(config.spacy_model)
        self.stop_words = set(stopwords.words('english'))

        if config.custom_stopwords:
            self.stop_words.update(config.custom_stopwords)

    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single txt document"""
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('','', string.punctuation))

        doc = self.nlp(text)

        tokens = []

        for token in doc:
            if self.config.remove_stopwords and token.text in self.stop_words:
                continue
            if self.config.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        return ' '.join(tokens)
    

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess_text(text) for text in texts]

        