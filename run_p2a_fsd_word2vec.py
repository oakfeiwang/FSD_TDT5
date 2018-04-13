import data_handler
import general_functions
from inverted_index import InvertedIndex
from evaluation import evaluation
from word2vec import generate_word2vec_vector_array

import cPickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import logging
from gensim.models import word2vec
from sklearn import svm


if __name__ == "__main__":

    # ###
    # #  1 read data into dictionary
    # ###

    raw_data_dictionary = data_handler.read_data("./data")

    ###
    #  2 generate id list and text list from dictionary
    ###

    id_list, raw_text_list = data_handler.generate_lists_from_dictionary(raw_data_dictionary)

    ###
    #  3 stem texts and remove stopwords
    ###

    num_data = 20000
    stemmed_text_list_all = data_handler.stem_text_and_remove_stopwords(raw_text_list, "Krovetz")
    stemmed_text_list = stemmed_text_list_all[:num_data]

    ###
    #  4.1 generate tf-idf features to build the same inverted index as p2p_tfidf model
    ###

    tfidfVectorizer = TfidfVectorizer(min_df=3)
    tf_idf_vectors = tfidfVectorizer.fit_transform(stemmed_text_list)
    tf_idf_vectors_array = tf_idf_vectors.toarray()
    features = tfidfVectorizer.get_feature_names()

    ###
    #  4.2 generate self-trained word2vec model
    ###
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    num_features = 300    # Word vector dimensionality
    min_word_count = 3   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    print "Training model..."

    sentences = data_handler.generate_sentences_list_from_raw_text_list(raw_text_list[:num_data])
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    model.init_sims(replace=True)

    words_list = data_handler.generate_words_list_from_text_list(stemmed_text_list)
    word2vec_vector_array = generate_word2vec_vector_array(words_list, model, 300)

    ###
    #  5 first story detection
    ###

    novelty_score_list_for_evaluation = []
    with open("story_list", 'rb') as handle:
        index_list_for_evaluation = cPickle.loads(handle.read())

    for i in index_list_for_evaluation:
        new_doc = word2vec_vector_array[i]
        new_doc_norm = np.dot(new_doc, new_doc)
        if new_doc_norm == 0:
            novelty_score = -1
        else:
            clf = svm.OneClassSVM(nu=0.1)
            if i > 2000:
                start_id = i - 2000
            else:
                start_id = 0
            clf.fit(word2vec_vector_array[start_id:i])
            novelty_score = -(clf.decision_function(new_doc.reshape(1, -1))[0][0])
        novelty_score_list_for_evaluation.append(novelty_score)

    min_novelty_score = min(novelty_score_list_for_evaluation)
    max_novelty_score = max(novelty_score_list_for_evaluation)

    ###
    #  6 evaluation
    ###
    with open('true_label_list', 'rb') as handle:
        true_labels = cPickle.loads(handle.read())

    plt.plot([0.0, 1.0], [1.0, 0.0], label="Random")

    false_alarms = []
    misses = []
    # scan the novelty scores with a range of thresholds
    thresholds = np.arange(min_novelty_score, max_novelty_score + 0.00001, 0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list_for_evaluation]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels, true_labels)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses, label="p2a_fsd_word2vec")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()