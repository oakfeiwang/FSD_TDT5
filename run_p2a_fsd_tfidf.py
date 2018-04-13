import data_handler
import general_functions
from evaluation import evaluation
import cPickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
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
    #  4 generate tf-idf features
    ###

    tfidfVectorizer = TfidfVectorizer(min_df=3, max_features=300)
    tf_idf_vectors = tfidfVectorizer.fit_transform(stemmed_text_list)
    tf_idf_vectors_array = tf_idf_vectors.toarray()
    features = tfidfVectorizer.get_feature_names()

    ###
    #  5 first story detection
    ###

    novelty_score_list_for_evaluation = []
    with open("story_list", 'rb') as handle:
        index_list_for_evaluation = cPickle.loads(handle.read())

    for i in index_list_for_evaluation:
        new_doc = tf_idf_vectors_array[i]
        new_doc_norm = np.dot(new_doc, new_doc)
        if new_doc_norm == 0:
            novelty_score = -1
        else:
            clf = svm.OneClassSVM(nu=0.1)
            if i > 2000:
                start_id = i-2000
            else:
                start_id = 0
            clf.fit(tf_idf_vectors_array[start_id:i])
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
    plt.step(false_alarms, misses, label="p2a_fsd_tfidf")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()