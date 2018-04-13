from collections import defaultdict, deque

class InvertedIndex():
    """
    Class of Inverted Index for P2P models
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        # the inverted index is stored in a dictionary with a word as key and a fixed-length list of story ids as value
        dq = lambda: deque(maxlen=maxlen)
        self.inverted_index_dictionary = defaultdict(dq)

    def search_and_index(self, words, new_doc_id):
        """ the function of searching from the inverted index and indexing the new data
        :param words: list of words from a story to be searched by
        :param new_doc_id: the id of the story
        :return: a list of story ids
        """
        document_ids = []
        for word in words:
            document_ids_ = self.inverted_index_dictionary[word]
            document_ids = list(set(list(document_ids) + list(document_ids_)))
            self.inverted_index_dictionary[word].append(new_doc_id)
        document_ids = sorted(document_ids)
        if len(document_ids) > self.maxlen:
            return document_ids[:-(self.maxlen + 1):-1]
        else:
            return document_ids
