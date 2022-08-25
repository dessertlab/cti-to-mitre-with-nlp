# cti-to-mitre-with-nlp
In this repo we provide a replication package for the paper "Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study". 

In the paper, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.

The project includes scripts and data to repeat the training/testing experiments on classification models presented in the paper.

# Project Organization

The diagram below provides the organization of the project:

```
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- bulid_dataset
|   -- prepare_dataset_data.py
|-- Colab_notebooks
|   -- finetune_secBert.ipynb
|   -- trained_secBert.ipynb
|-- data
|   -- dataset.csv
|   -- other_datasets
|      -- dataset_mixed.csv
|      -- dataset_tram.csv
|-- src
|   -- bar_plot_documents.py
|   -- CNN_clf.py
|   -- document_analysis.py
|   -- document_data.py
|   -- LSTM_clf.py
|   -- ml_classifier.py
|   -- pretrained_embedding_LSTM.py
|   -- deepl_utils.py
|   -- utils
|   -- apt_documents
|      -- FIN6
|      -- MenuPass
|      -- WizardSpider
```
To be able to correctly run all project' scripts, download the addtional files from https://figshare.com/articles/dataset/additional_files_zip/20076281  and unzip the archive into the main directory, extract the inner directories and delete the root. You will find:
```
|-- cti
|   -- capec
|      -- 2.0
|   -- enterprise_attack
|      -- enterprise-attack-10.1.json
|-- model
```
In the `model` directory you will find the pretrained Word2vec model provided by https://ebiquity.umbc.edu/resource/html/id/379/Cybersecurity-embeddings that was used as a baseline to bulid the LSTM model starting from a pretrained word2vec. 

In the `build_dataset` directory you can find the script to automatically generate the dataset from MITRE ATT&CK and CAPEC Knowledge bases, distributed in the STIX format. Notice that we loaded into the external repo the KB used in our work (see directory `cti` above), but you can pontentially download a newer version from official repositories (ATT&CK -> https://github.com/mitre-attack/attack-stix-data, CAPEC -> https://github.com/mitre/cti) and update the dataset. 

In the `data` directory you can find the dataset.csv file. It contains the dataset used for the training and evaluation of our classification models. In the `other_datasets` sub-directory there are additional datasets used for the validation of our preliminary results. 

In the `Colab_notebooks` directory you will find the code to finetune a BERT model trained on cybersecurity terms (https://github.com/jackaduma/SecBERT) and another to repeat all the analysis: calculate the metrics and performing the document analysis on such model. 

Finally, in the `src` directory you can find all the scripts: the ones used for the training/testing of our classification models and the ones employed for the document analysis. In the `src/apt_documents/FIN6`, `src/apt_documents/WizardSpider`, and `src/apt_documents/MenuPass` sub-directories you can find the resources chosen and employed in the document analysis. 

# Installation steps 

## Install the dependencies 
```
cd cti-to-mitre-with-nlp
pip install -r requirements.txt
```
## Download the zip file 
To be able to correctly run all project' scripts, download the addtional files from https://figshare.com/articles/dataset/additional_files_zip/20076281 and unzip the archive into the main directory, extract the inner directories and delete the root. See [Project Organization](#project-organization) details.


## Troubleshooting 
If you get an error like this:

```
  Resource punkt not found.

  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk

  >>> nltk.download('punkt')

  For more information see: https://www.nltk.org/data.html

  Attempted to load tokenizers/punkt/PY3/english.pickle
```
  
  Try  
  ```
  python -m nltk.downloader <RESOURCE_NAME>
  ```
  es.  
  ```
  python -m nltk.downloader punkt
  ```
  
  # A guide through the scripts 
  
  All the main scripts are in the `src` directory: 
  
  - *ml_classifier.py* provides an implementation of different traditional ML classifiers. The training and testing tasks are performed, metrics showed and trained model saved. The script produces in output a directory `ml_models` where all classifiers' model are stored. 
  - *LSTM_clf.py*, *CNN_clf.py* and *pretrained_embedding_LSTM.py* present the same structure. They use the classes provided by *deepl_utils.py* to build a particular DL classifier, compute the metrics and save the model. For each model a directory `<MODEL_NAME>\_model` is created: it contains the tokenizer and model files. 
  - *document_data.py* gives the necessary information to conduct the document analysis. For each document a list of associated techniques is here provided. 
  - *document_analysis.py* performs the analysis on chosen documents reloading classifiers' models (you need to run the classifiers' scripts first) and evaluating the correctness of the predictions. 
  - *bar_plot_documents.py* generates graphs from the document analysis results and save them into the documents' paths.

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
