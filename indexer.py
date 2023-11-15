'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from .linkedlist import LinkedList
from collections import OrderedDict


class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})
        self.total_docs = set()

    def get_index(self):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index
            Already implemented."""
        tokens_in_doc = len(list(tokenized_document))
        for token in tokenized_document:
            token_freq = tokenized_document.count(token)
            tf = token_freq/tokens_in_doc
            self.add_to_index(token, doc_id, tf)

    def add_to_index(self, term_, doc_id_,tf):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the posstings list of the term.
            To be implemented."""
        # raise NotImplementedError
        if term_ in self.inverted_index:
            self.inverted_index[term_].insert_at_end(doc_id_, tf)
            self.total_docs.add(doc_id_)
        else:
            llist = LinkedList()
            llist.insert_at_end(doc_id_, tf)
            self.inverted_index[term_] = llist
            self.total_docs.add(doc_id_)

    def sort_terms(self):
        """ Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers.
            To be implemented."""
        # raise NotImplementedError
        for term, _ in self.inverted_index.items():
            self.inverted_index[term].add_skip_connections()

    def calculate_tf_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index.
            To be implemented."""
        # raise NotImplementedError
        tokens = list(self.inverted_index.keys())
        for token in tokens:
                doc = self.inverted_index[token]
                posting_list_length = self.inverted_index[token].length      
                idf = len(self.total_docs)/posting_list_length
                if doc is not None:
                    doc = doc.start_node
                    while doc is not None:
                        doc.tf_idf = doc.tf_idf * idf
                        doc = doc.next     