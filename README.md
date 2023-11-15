- Implemented an Inverted Index data structure using Python3 to map tokens to their corresponding document IDs within a corpus.

- Engineered a preprocessing pipeline to transform raw text data into a searchable index, including case normalization, special character removal, whitespace trimming, tokenization, stop word removal, and Porter's stemming.

- Developed a Document-at-a-Time (DAAT) search strategy to process Boolean "AND" queries against the index.

- Constructed posting lists with skip pointers to enhance the efficiency of query processing, improving search performance through optimized comparison operations.

- Incorporated tf-idf scoring into the index to sort documents by relevance, calculating term frequency and inverse document frequency for each token within the corpus.

- Exposed the search functionality through a Flask API endpoint, enabling external applications to query the index.
