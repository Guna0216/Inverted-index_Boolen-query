'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from tqdm import tqdm
from .preprocessor import Preprocessor
from .indexer import Indexer
from collections import OrderedDict
from .linkedlist import LinkedList
import inspect as inspector
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib

app = Flask(__name__)


class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def _merge(self, list1, list2, skip = False):
        """ Implement the merge algorithm to merge 2 postings list at a time.
            Use appropriate parameters & return types.
            While merging 2 postings list, preserve the maximum tf-idf value of a document.
            To be implemented."""
        # raise NotImplementedError
        if skip:
            new_list = LinkedList()
            list1_node = list1.start_node
            list2_node = list2.start_node
            comparisions = 0
            while list1_node is not None and list2_node is not None:
                if list1_node.value == list2_node.value:
                    tf = max(list1_node.tf_idf, list2_node.tf_idf)
                    new_list.insert_at_end(list1_node.value,tf) 
                    list1_node = list1_node.next
                    comparisions += 1
                elif list1_node.value < list2_node.value:
                    if (list1_node.skip is not None) and (list1_node.skip.value <= list2_node.value):
                        while (list1_node.skip is not None) and (list1_node.skip.value <= list2_node.value):
                            list1_node = list1_node.skip
                            comparisions = comparisions + 1
                    else:
                        list1_node = list1_node.next
                        comparisions += 1
                else:
                    if (list2_node.skip is not None) and (list2_node.skip.value <= list1_node.value):
                        while (list2_node.skip is not None) and (list2_node.skip.value <= list1_node.value):
                            list2_node = list2_node.skip
                            comparisions = comparisions + 1
                    else:
                        list2_node = list2_node.next
                        comparisions += 1
                new_list.add_skip_connections()
            return new_list, comparisions
        else:
            new_list = LinkedList()
            list1_node = list1.start_node
            list2_node = list2.start_node
            comparisions = 0
            while list1_node is not None and list2_node is not None:
                if list1_node.value == list2_node.value:
                    tf = max(list1_node.tf_idf, list2_node.tf_idf)
                    new_list.insert_at_end(list1_node.value,tf)
                    list1_node = list1_node.next
                    comparisions += 1
                elif list1_node.value < list2_node.value:
                    list1_node = list1_node.next
                    comparisions += 1
                else:
                    list2_node = list2_node.next
                    comparisions += 1
            new_list.add_skip_connections()
            return new_list, comparisions        
        
    def _daat_and(self, query, skip = False,tfidf_sort = False):
        """ Implement the DAAT AND algorithm, which merges the postings list of N query terms.
            Use appropriate parameters & return types.
            To be implemented."""
        # raise NotImplementedError
        tokens = query
        if skip:
            new_list = LinkedList()
            if self.indexer.get_index()[tokens[0]].length <= self.indexer.get_index()[tokens[1]].length:
                new_list, total_comparisions = self._merge(self.indexer.get_index()[tokens[0]],self.indexer.get_index()[tokens[1]], skip = True)
            else:
                new_list, total_comparisions = self._merge(self.indexer.get_index()[tokens[1]],self.indexer.get_index()[tokens[0]], skip = True)
            counter = 2
            while counter < len(tokens):
                if new_list.length <= self.indexer.get_index()[tokens[counter]].length:
                    new_list, comparisions = self._merge(new_list, self.indexer.get_index()[tokens[counter]], skip = True)
                else:
                    new_list, comparisions = self._merge(self.indexer.get_index()[tokens[counter]], new_list, skip = True)
                total_comparisions += comparisions
                counter += 1
        else:
            new_list = LinkedList()
            if self.indexer.get_index()[tokens[0]].length <= self.indexer.get_index()[tokens[1]].length:
                new_list, total_comparisions = self._merge(self.indexer.get_index()[tokens[0]],self.indexer.get_index()[tokens[1]])
            else:
                new_list, total_comparisions = self._merge(self.indexer.get_index()[tokens[1]],self.indexer.get_index()[tokens[0]])
            counter = 2
            while counter < len(tokens):
                if new_list.length <= self.indexer.get_index()[tokens[counter]].length:
                    new_list, comparisions = self._merge(new_list, self.indexer.get_index()[tokens[counter]])
                else:
                    new_list, comparisions = self._merge(self.indexer.get_index()[tokens[counter]], new_list)
                total_comparisions += comparisions
                counter += 1
        
        if tfidf_sort:
            doc_tfidf = []
            doc = new_list.start_node
            while doc:
                doc_tfidf.append((doc.value,doc.tf_idf))
                doc = doc.next
            doc_tfidf = sorted(doc_tfidf, key = lambda x: x[1], reverse = True)
            doc_sorted = []
            for ele in doc_tfidf:
                doc_sorted.append(ele[0])
            return doc_sorted, total_comparisions
        else:
            return new_list.traverse_list(), total_comparisions

    def _get_postings(self,token, skip = False):
        """ Function to get the postings list of a term from the index.
            Use appropriate parameters & return types.
            To be implemented."""
        # raise NotImplementedError
        if skip:
            return self.indexer.get_index()[token].traverse_skips() 
        else:
            return self.indexer.get_index()[token].traverse_list()

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change."""
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        with open(corpus, 'r') as fp:
            for line in fp.readlines():
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list, random_command):
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       'sanity': self.sanity_checker(random_command)}

        for query in tqdm(query_list):
            """ Run each query against the index. You should do the following for each query:
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""
            # raise NotImplementedError
            input_term_arr = self.preprocessor.tokenizer(query)  # Tokenized query. To be implemented.

            for term in input_term_arr:
                postings, skip_postings = self._get_postings(term), self._get_postings(term, skip = True)

                """ Implement logic to populate initialize the above variables.
                    The below code formats your result to the required format.
                    To be implemented."""

                output_dict['postingsList'][term] = postings
                output_dict['postingsListSkip'][term] = skip_postings

            and_op_no_skip = self._daat_and(input_term_arr,skip = False, tfidf_sort = False)[0]
            and_op_skip = self._daat_and(input_term_arr,skip = True, tfidf_sort = False)[0]
            and_op_no_skip_sorted = self._daat_and(input_term_arr,skip = False, tfidf_sort = True)[0]
            and_op_skip_sorted = self._daat_and(input_term_arr,skip = True, tfidf_sort = True)[0]
            and_comparisons_no_skip = self._daat_and(input_term_arr,skip = False, tfidf_sort = False)[1]
            and_comparisons_skip = self._daat_and(input_term_arr,skip = True, tfidf_sort = False)[1]
            and_comparisons_no_skip_sorted = self._daat_and(input_term_arr,skip = False, tfidf_sort = True)[1]
            and_comparisons_skip_sorted = self._daat_and(input_term_arr,skip = True, tfidf_sort = True)[1]
            """ Implement logic to populate initialize the above variables.
                The below code formats your result to the required format.
                To be implemented."""
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_op_no_skip)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_op_skip)
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(and_op_no_skip_sorted)
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(and_op_skip_sorted)

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = and_op_no_score_no_skip
            output_dict['daatAnd'][query.strip()]['num_docs'] = and_results_cnt_no_skip
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = and_comparisons_no_skip

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = and_op_no_score_skip
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = and_results_cnt_skip
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = and_comparisons_skip

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = and_op_no_score_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = and_results_cnt_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_no_skip_sorted

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = and_op_no_score_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = and_results_cnt_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_skip_sorted

        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
        Do NOT change it."""
    start_time = time.time()

    queries = request.json["queries"]
    random_command = request.json["random_command"]

    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries, random_command)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
        Do NOT change it."""

    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.")
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", type=str,
                        help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
                             "DO NOT pass incorrect value here")

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    """ Initialize the project runner"""
    runner = ProjectRunner()

    """ Index the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    runner.run_indexer(corpus)

    app.run(host="0.0.0.0", port=9999)
