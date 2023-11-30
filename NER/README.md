## Mountain names recognition

Abstract: This repo contains a model for the identification of
mountain names inside the texts. 

### About dataset
The dataset was created from different papers about mountains like [this](https://edubirdie.com/examples/mountains/) and [this](https://www.memphistours.com/blog/12-most-famous-mountains-in-the-world). Chat-GPT helped me with finding mountain names.
Also, I added some data from this Kaggle data set [link](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)
The final dataset is created using the file "NER_dataset_creation.ipynb", and resides in the file "dataset.csv"


### Guide

1. We need to create a base directory and download the files from the repository there.

2. Install libraries from requirements.txt

3. Train your model, by running the file train.py.

4. To use the model, you need to run the test.py file.

