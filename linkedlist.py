'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import math


class Node:

    def __init__(self, value=None, next=None,tf= 0):
        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Add more parameters if needed.
            Hint: You may want to define skip pointers & appropriate score calculation here"""
        self.value = value
        self.next = next
        self.skip = None
        self.tf_idf = tf


class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
        Feel free to add additional functions to this class."""
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        if self.start_node is None:
            return
        else:
            """ Write logic to traverse the linked list.
                To be implemented."""
            # raise NotImplementedError
            current = self.start_node    
            traversal.append(current.value)
            while current.next:
                current = current.next
                traversal.append(current.value)
            return traversal

    def traverse_skips(self):
        traversal = []
        if self.start_node is None:
            return
        elif self.start_node.skip is None:
            return traversal
        else:
            """ Write logic to traverse the linked list using skip pointers.
                To be implemented."""
            # raise NotImplementedError
            current = self.start_node
            while current is not None:
                traversal.append(current.value)
                current = current.skip
            return traversal

    def add_skip_connections(self):
        n_skips = math.floor(math.sqrt(self.length))
        if n_skips * n_skips == self.length:
            n_skips = n_skips - 1
        """ Write logic to add skip pointers to the linked list. 
            This function does not return anything.
            To be implemented."""
        # raise NotImplementedError
        self.n_skips = n_skips
        self.skip_length = int(round(math.sqrt(self.length)))
        current = self.start_node
        skips = 0
        if self.length <= self.n_skips:
            return
        else:
            while skips < self.n_skips:
                skip_length = 0
                skip_node = current
                while skip_length < self.skip_length:
                    skip_node = skip_node.next
                    skip_length += 1
                current.skip = skip_node
                current = skip_node
                skips += 1   

    def insert_at_end(self, value, tf):
        """ Write logic to add new elements to the linked list.
            Insert the element at an appropriate position, such that elements to the left are lower than the inserted
            element, and elements to the right are greater than the inserted element.
            To be implemented. """
        # raise NotImplementedError
        current = self.start_node
        while current:
            if current.value == value:
                current.tf = tf
                return
            current = current.next

        new_node = Node(value = value, tf = tf)
        if not self.start_node:
            self.start_node = new_node
            self.end_node = new_node
        else:
            current = self.start_node
            prev = None
            while current is not None:
                if current.value >= value:
                    break
                prev = current
                current = current.next

            if not prev:
                new_node.next = self.start_node
                self.start_node = new_node
            else:
                prev.next = new_node
                new_node.next = current

        self.length += 1

