import data_handler
import general_functions
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
    #  5 first story detection
    ###

    false_alarms = []
    misses = []

    with open('true_label_list', 'rb') as handle:
        true_labels = cPickle.loads(handle.read())

    for threshold in np.arange(0.0, 1.05, 0.05):

        predicted_labels = []
        predicted_labels_for_evaluation = []

        centroid_list = []
        sum_list = []
        amount_list = []

        for i in range( tf_idf_vectors_array.shape[0]):
            novelty_score = 1
            cluster = None
            predicted_label = False

            new_doc =  tf_idf_vectors_array[i]
            new_doc_norm = np.dot(new_doc, new_doc)
            if new_doc_norm == 0:
                predicted_label = False
            else:
                for j in range(len(centroid_list)):
                    dist = general_functions.cosine_distance(new_doc, centroid_list[j])
                    if dist < novelty_score:
                        novelty_score = dist
                        cluster = j
                if novelty_score > threshold:
                    centroid_list.append(new_doc)
                    sum_list.append(new_doc)
                    amount_list.append(1)
                    predicted_label = True
                else:
                    sum_list[cluster] = np.add(sum_list[cluster], new_doc)
                    amount_list[cluster] += 1
                    centroid_list[cluster] = np.divide(sum_list[cluster], amount_list[cluster])
                    predicted_label = False

            predicted_labels.append(predicted_label)

        with open("story_list", 'rb') as handle:
            index_list_for_evaluation = cPickle.loads(handle.read())

        for i, key in enumerate(id_list):
            if key in index_list_for_evaluation:
                predicted_labels_for_evaluation.append(predicted_labels[i])

        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels_for_evaluation,true_labels)

        false_alarms.append(false_alarm)
        misses.append(miss)

    ###
    #  6 generate figure
    ###

    plt.plot([0.0, 1.0], [1.0, 0.0], label="Random")
    plt.step(false_alarms, misses, label="p2c_fsd_tfidf")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()








