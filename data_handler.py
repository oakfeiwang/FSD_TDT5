""" Functions for data maniputlation """

import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from krovetzstemmer import Stemmer
import string
import nltk
#nltk.download()

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def read_data(datadir):
    """ Read TDT5 data from a directory
    :param datadir: the address of the directory containing tdt5 data
    :return: a dictionary with document ids as keys and document text as values
    """

    data_dictionary = {}
    for parent, dirnames, filenames in os.walk(datadir):
        for filename in filenames:
            with open(os.path.join(parent, filename), 'r') as f:
                data = f.read()
            soup = BeautifulSoup(data, "html.parser")
            docnos = soup.findAll('docno')
            texts = soup.findAll('text')
            length = len(docnos)
            ###
            # the document id is filename concatenated by document text with only letters and numbers
            ###
            filename_ = re.sub('[^a-zA-Z0-9]', '', filename.encode("utf-8").strip())[:19]
            for i in range(length):
                docno = re.sub('[^a-zA-Z0-9]', '', docnos[i].text.encode("utf-8").strip())
                doc_id = filename_ + docno
                doc_text = texts[i].text.encode("utf-8").strip()
                data_dictionary[doc_id] = doc_text
    return data_dictionary

def generate_lists_from_dictionary(data_dictionary):
    """ generate sorted id list and text list from dictionary
    :param data_dictionary: the data dictionary
    :return: sorted id list, sorted text list
    """
    text_list = []
    id_list = []
    for key in sorted(data_dictionary.iterkeys()):
        id_list.append(key)
        text_list.append(data_dictionary[key])
    return id_list, text_list

def tokenise_text(text):
    """ tokenise text to word list and remove the items with purely punctuation
    :param text: text to be tokenised
    :return: word list without items with purely punctuation
    """
    punctuations = list(string.punctuation)
    punctuations.append("\'\'")
    punctuations.append("`")
    punctuations.append("``")
    punctuations.append("--")
    punctuations.append("//")
    punctuations.append("///")

    words = [t for t in word_tokenize(text) if t not in punctuations]

    return words

def stem_text_and_remove_stopwords(text_list, stemmer_type="Porter"):
    """ stem text and remove stopwords
    :param text_list: the texts to be stemmed
    :param stemmer_type: the type of stemmers: "Porter" stemmer or "Krovetz" stemmer
    :return: stemmed text list without stopwords
    """

    if stemmer_type == "Porter":
        stemmer = PorterStemmer()
    elif stemmer_type == "Krovetz":
        stemmer = Stemmer()
    else:
        print "wrong stemmer type!!!"

    stemmed_text_list = []
    for text in text_list:
        words = tokenise_text(text)
        for idx, w in enumerate(words):
            words[idx] = stemmer.stem(w.decode("utf-8", "ignore"))
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        stemmed_text_list.append(" ".join(words))

    return stemmed_text_list

def generate_words_list_from_text_list(text_list):
    """ convert list of words from list of texts to be converted
    :param text_list: list of texts
    :return: list of words
    """
    words_list = []
    for text in text_list:
        words_list.append(tokenise_text(text))
    return words_list

def data_to_sentences(datadir):
    """ generate list of sentences from raw data for the training of Word2Vec
    :param datadir: the address of the directory containing tdt5 data
    :return: list of sentences
    """
    sentences = []
    j=0
    for parent,dirnames,filenames in os.walk(datadir):
        for filename in filenames:
            j=j+1
            print j
            with open(os.path.join(parent,filename), 'r') as f:
                data= f.read()
            soup = BeautifulSoup(data)
            ids = soup.findAll('docno')
            contents = soup.findAll('text')
            if len(contents) == len(ids):
                for i in range(len(contents)):
                    sentences += data_to_wordsentences(contents[i])
    return sentences

def data_to_wordsentences(raw_data):
    """ convert a text to list of sentences
    :param raw_data: a text to be converted
    :return: list if sentences
    """
    sentences = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_data.text.strip())
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words = tokenise_text(raw_sentence)
            stemmer = Stemmer()
            for idx, w in enumerate(words):
                words[idx] = stemmer.stem(w.decode("utf-8", "ignore"))
            sentences.append(words)
    return sentences

def generate_sentences_list_from_raw_text_list(raw_text_list):
    """ convert list of texts into list of sentences for the traning of Word2Vec
    :param raw_text_list: list of texts to be converted
    :return: list if sentences
    """

    sentences_list = []
    stemmer = Stemmer()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for i in range(len(raw_text_list)):
        raw_sentences = tokenizer.tokenize(raw_text_list[i])
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                words = tokenise_text(raw_sentence)
                for idx, w in enumerate(words):
                    words[idx] = stemmer.stem(w.decode("utf-8", "ignore"))
                sentences_list.append(words)
    return sentences_list