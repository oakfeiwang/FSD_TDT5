import cPickle
import numpy as np
import matplotlib.pyplot as plt

def evaluation(pred_labels, true_labels):

    if len(pred_labels) == len(true_labels):
        pred_labels = np.asarray(pred_labels)
        true_labels = np.asarray(true_labels)
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

        if (FP+TN)!=0:
            false_alarm = float(FP)/(FP+TN)
        else:
            false_alarm = None

        if (TP + FN)!=0:
            miss = float(FN)/(TP + FN)
        else:
            miss = None

    else:
        print "the length of pre_labels and true_labels are not equal!"
    return TP, TN, FP, FN, false_alarm, miss

if __name__ == "__main__":

    with open('true_label_list_first_20000', 'rb') as handle:
        evaluation_true_labels = cPickle.loads(handle.read())

    plt.plot([0.0,1.0], [1.0,0.0],label="Random")

    with open("./results_for_p2p_tfidf/with_all_features/novelty_score_list_for_evaluation_first_20000", 'rb') as handle:
        novelty_score_list = cPickle.loads(handle.read())
    false_alarms = []
    misses = []
    cost_functions = []
    thresholds = np.arange(0.00000,1.00001,0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels,evaluation_true_labels)
        # result= "threshold="+str(threshold)+", TP="+str(TP)+", TN="+str(TN)+", FP="+str(FP)+", FN="+str(FN)+", false alarm="+str(false_alarm)+", miss="+str(miss)+", cost function="+str(cost_function)+"\n"
        # with open("results.txt", "a") as handle:
        #     handle.write(result)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses,label="p2p_fsd_tfidf")

    with open("./results_for_p2p_word2vec/self_trained/novelty_score_list_for_evaluation_first_20000", 'rb') as handle:
        novelty_score_list = cPickle.loads(handle.read())
    false_alarms = []
    misses = []
    cost_functions = []
    thresholds = np.arange(0.00000,1.00001,0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels,evaluation_true_labels)
        # result= "threshold="+str(threshold)+", TP="+str(TP)+", TN="+str(TN)+", FP="+str(FP)+", FN="+str(FN)+", false alarm="+str(false_alarm)+", miss="+str(miss)+", cost function="+str(cost_function)+"\n"
        # with open("results.txt", "a") as handle:
        #     handle.write(result)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses,label="p2p_fsd_word2vec")



    false_alarms_tfidf = [0.983193277311,0.878151260504,0.810924369748,0.743697478992,0.668067226891,0.613445378151,0.495798319328,0.386554621849,0.289915966387,0.201680672269,0.121848739496,0.0756302521008,0.0420168067227,0.0252100840336,0.0,0.00420168067227,0.00840336134454,0.0]
    misses_tfidf = [0.0,0.0,0.0,0.0,0.0555555555556,0.0555555555556,0.111111111111,0.166666666667,0.166666666667,0.222222222222,0.277777777778,0.333333333333,0.333333333333,0.333333333333,0.611111111111,0.833333333333,0.944444444444,1.0]
    plt.step(false_alarms_tfidf, misses_tfidf, label="p2c_fsd_tfidf")

    false_alarms_word2vec = [1.0,0.86974789916,0.726890756303,0.609243697479,0.542016806723,0.470588235294,0.386554621849,0.319327731092,0.264705882353,0.22268907563,0.189075630252,0.138655462185,0.100840336134,0.0882352941176,0.0588235294118,0.063025210084,0.0588235294118,0.0420168067227,0.046218487395,0.0378151260504,0.0378151260504,0.0294117647059, 0.0252100840336,0.0252100840336,0.0252100840336,0.0294117647059,0.00840336134454,0.00420168067227]
    misses_word2vec = [0.0,0.0,0.0555555555556,0.111111111111,0.111111111111,0.111111111111,0.111111111111,0.166666666667,0.277777777778,0.333333333333,0.333333333333,0.388888888889,0.444444444444,0.5,0.5,0.444444444444,0.555555555556,0.555555555556,0.611111111111,0.722222222222,0.666666666667,0.666666666667,0.777777777778,0.722222222222,0.777777777778,0.833333333333,0.888888888889,1.0]
    plt.step(false_alarms_word2vec, misses_word2vec, label="p2c_fsd_word2vec")

    with open("./results_for_p2a_tfidf/novelty_score_list_for_evaluation_first_20000", 'rb') as handle:
        novelty_score_list = cPickle.loads(handle.read())
    false_alarms = []
    misses = []
    cost_functions = []
    thresholds = np.arange(-0.05500,0.01801,0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels,evaluation_true_labels)
        # result= "threshold="+str(threshold)+", TP="+str(TP)+", TN="+str(TN)+", FP="+str(FP)+", FN="+str(FN)+", false alarm="+str(false_alarm)+", miss="+str(miss)+", cost function="+str(cost_function)+"\n"
        # with open("results.txt", "a") as handle:
        #     handle.write(result)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses,label="p2a_fsd_tfidf")


    with open("./results_for_p2a_word2vec/novelty_score_list_for_evaluation_first_20000", 'rb') as handle:
        novelty_score_list = cPickle.loads(handle.read())
    false_alarms = []
    misses = []
    cost_functions = []
    thresholds = np.arange(-0.06300,0.12371,0.00001)
    for threshold in thresholds:
        predicted_labels = [novelty_score > threshold for novelty_score in novelty_score_list]
        TP, TN, FP, FN, false_alarm, miss = evaluation(predicted_labels,evaluation_true_labels)
        # result= "threshold="+str(threshold)+", TP="+str(TP)+", TN="+str(TN)+", FP="+str(FP)+", FN="+str(FN)+", false alarm="+str(false_alarm)+", miss="+str(miss)+", cost function="+str(cost_function)+"\n"
        # with open("results.txt", "a") as handle:
        #     handle.write(result)
        false_alarms.append(false_alarm)
        misses.append(miss)
    plt.step(false_alarms, misses,label="p2a_fsd_word2vec")

    plt.legend()
    plt.xlabel('false alarm rate')
    plt.ylabel('missing rate')
    plt.show()