import numpy as np


def generate_word2vec_vector(words, model, num_features):
    """ generate word2vec vector for a text
    :param words: a list of words of a text
    :param model: a trained Word2Vec model
    :param num_features: number of features of the output Word2Vec feature
    :return: a Word2Vec vector
    """
    word2vec_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words = num_words + 1.
            word2vec_vector = np.add(word2vec_vector, model[word])
    if num_words > 0:
        word2vec_vector = np.divide(word2vec_vector, num_words)
    return word2vec_vector


def generate_word2vec_vector_array(words_list, model, num_features):
    """ generate a list of word2vec vectors for a list of texts
    :param words_list: a list of word lists
    :param model: a trained Word2Vec model
    :param num_features: number of features of the output Word2Vec feature
    :return: a list of Word2Vec vectors
    """
    count = 0.
    word2vec_vector_array = np.zeros((len(words_list), num_features), dtype="float32")

    for words in words_list:
        word2vec_vector_array[int(count)] = generate_word2vec_vector(words, model, num_features)
        count = count + 1.
    return word2vec_vector_array