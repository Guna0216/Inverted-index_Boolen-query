'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import collections
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
            Already implemented"""
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """ Implement logic to pre-process & tokenize document text.
            Write the code in such a way that it can be re-used for processing the user's query.
            To be implemented."""
        # raise NotImplementedError
        stopwords = nltk.corpus.stopwords.words('english')
        text = text.lower()
        text = re.sub(r'-',' ',text)
        text = re.sub(r'–',' ',text)
        text = re.sub(r':','',text)
        text = re.sub(r'/',' ',text)
        text = re.sub(r'—', ' ',text)
        text = re.sub(r'‐', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        text = text.split()
        text = [ele for ele in text if ele not in stopwords]
        text = [self.ps.stem(ele) for ele in text]
        return text




