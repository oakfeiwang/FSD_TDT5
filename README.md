# fsd_tdt5
First Story Detection with TDT5 dataset in the paper - Exploring Online Novelty Detection Using First Story Detection Models.

## Software requirements

* Python 2
* NLTK
* gensim

## Data setup

The TDT5 data is available for downloading from LDC at https://catalog.ldc.upenn.edu/ldc2006t18. After downloading the data, create a new folder `/data` in the `working directory`, and extract all the English data files (with _ENG in the name) from the folder `tdt5_mltxt_v1_1_LDC2006T18.tgz\tdt5_mltxt_v1_1_LDC2006T18.tar\tdt5_mltxt_v1_1\data\tkn_sgm` into the newly created folder;

The TDT5 annotated labels are available for downloading from LDC at https://catalog.ldc.upenn.edu/LDC2006T19. After data downloads, extract the file `TDT2004.topic_rel.v2.0` from the folder `LDC2006T19.tgz\tdt5_topic_annot.tar\tdt5_topic_annot\data\annotations\topic_relevance\` into the `working directory`.

## Running the experiments

There are six experiments in the paper - Exploring Online Novelty Detection Using First Story Detection Models. The corresponding Python files are as follows:

* `run_p2p_fsd_tfidf.py`
* `run_p2p_fsd_word2vec.py`
* `run_p2c_fsd_tfidf.py`
* `run_p2c_fsd_word2vec.py`
* `run_p2a_fsd_tfidf.py`
* `run_p2a_fsd_word2vec.py`

Run `ground_truth.py` before the experiments.

These Python files can be run by 
```
python filename
```






