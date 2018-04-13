import data_handler
import general_functions
from evaluation import evaluation
from word2vec import generate_word2vec_vector_array

import cPickle
import numpy as np
import matplotlib.pyplot as plt
import logging
from gensim.models import word2vec


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
    #  4 generate self-trained word2vec model
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

    false_alarms = []
    misses = []

    with open('true_label_list', 'rb') as handle:
        true_labels = cPickle.loads(handle.read())

    for threshold in np.arange(0.0, 2.05, 0.05):

        predicted_labels = []
        predicted_labels_for_evaluation = []

        centroid_list = []
        sum_list = []
        amount_list = []

        for i in range( word2vec_vector_array.shape[0]):
            novelty_score = 1
            cluster = None
            predicted_label = False

            new_doc =  word2vec_vector_array[i]
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
    plt.step(false_alarms, misses, label="p2c_fsd_word2vec")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()





