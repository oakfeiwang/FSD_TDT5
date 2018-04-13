import data_handler
import general_functions
from inverted_index import InvertedIndex
from evaluation import evaluation
import cPickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


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
    #  4 generate tf-idf features
    ###

    tfidfVectorizer = TfidfVectorizer(min_df=3)
    tf_idf_vectors = tfidfVectorizer.fit_transform(stemmed_text_list)
    tf_idf_vectors_array = tf_idf_vectors.toarray()
    features = tfidfVectorizer.get_feature_names()

    ###
    #  5 inverted index
    ###

    max_comparison = 2000
    inverted_index = InvertedIndex(maxlen=max_comparison)

    ###
    #  6 first story detection
    ###

    novelty_score_list_for_evaluation = []
    nearest_neighbour_list_for_evaluation = []

    for i in range(tf_idf_vectors_array.shape[0]):
        # the range of novelty_score for tfidf is [0,1]
        novelty_score = 1
        nearest_neighbour = None

        # get the indices of words that are not 0 in tf-idf
        indices = tf_idf_vectors[i].indices[tf_idf_vectors[i].indptr[0]:tf_idf_vectors[i].indptr[1]]
        # get the words from the indices
        words = [features[idx] for idx in indices]
        # search the document ids to compare from inverted index
        document_ids_to_compare = inverted_index.search_and_index(words, i)

        with open("story_list", 'rb') as handle:
            index_list_for_evaluation = cPickle.loads(handle.read())

        if i in index_list_for_evaluation:
            new_doc = tf_idf_vectors_array[i]
            new_doc_norm = np.dot(new_doc, new_doc)
            if new_doc_norm == 0:
                novelty_score = -1
            else:
                for id in document_ids_to_compare:
                    # pre_doc = tf_idf_vectors[id,:].toarray()[:, indices][0]
                    pre_doc = tf_idf_vectors_array[id]
                    pre_doc_norm = np.dot(pre_doc, pre_doc)
                    if pre_doc_norm > 0:
                        dist = general_functions.cosine_distance(new_doc, pre_doc)
                        if dist < novelty_score:
                            novelty_score = dist
            novelty_score_list_for_evaluation.append(novelty_score)

    ###
    #  7 evaluation
    ###
    with open('true_label_list', 'rb') as handle:
        true_labels = cPickle.loads(handle.read())

    plt.plot([0.0, 1.0], [1.0, 0.0], label="Random")

    false_alarms = []
    misses = []
    # scan the novelty scores with a range of thresholds
    thresholds = np.arange(0.00000, 1.00001, 0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list_for_evaluation]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels, true_labels)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses, label="p2p_fsd_tfidf")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()





