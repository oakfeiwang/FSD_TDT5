###
# Generate the list of ground truth labels and the story list, first_story_list for evaluation
###

import data_handler

import pickle
import cPickle
import re

if __name__ == "__main__":

    # read from the ground truth file
    label_filename = "./TDT2004.topic_rel.v2.0"
    with open(label_filename) as f:
        contents = f.readlines()
    contents = contents[1:-1]
    contents = [re.sub('[^A-Za-z0-9]', '', x.encode("utf-8").strip()) for x in contents]

    topic_id_dictionary = {}    #topic: ids
    for i in range(1,251):
        topic_id_dictionary[i] = []

    first_story_list = []
    story_list = []
    num_data = 20000

    raw_data_dictionary = data_handler.read_data("./data")
    id_list_all, raw_text_list_all = data_handler.generate_lists_from_dictionary(raw_data_dictionary)
    id_list = id_list_all[:num_data]

    count = 0
    for content in contents:
        if count%100 == 0:
            print count
        topic_id = int(content[16]+content[17]+content[18])
        document_id = ""
        for j in range(32,51):
            document_id += content[j]
        file_id = ""
        for j in range(57, 76):
            file_id +=content[j]
        id = file_id+document_id
        if id in id_list:
            topic_id_dictionary[topic_id].append(id)
        count += 1
    for i in range(1, 251):
        if len(topic_id_dictionary[i]) > 0:
            first_story_list.append(min(topic_id_dictionary[i]))
            for j in range(0, len(topic_id_dictionary[i])):
                story_list.append(topic_id_dictionary[i][j])


    first_story_list = sorted(list(set(first_story_list)))
    story_list = sorted(list(set(story_list)))

    with open("first_story_list", 'wb') as handle:
        cPickle.dump(first_story_list, handle)
    with open("story_list", 'wb') as handle:
        cPickle.dump(story_list, handle)

    true_label_list = []
    for key in story_list:
        if key in first_story_list:
            true_label_list .append(True)
        else:
            true_label_list .append(False)

    with open("true_label_list", 'wb') as handle:
        cPickle.dump(true_label_list, handle)

