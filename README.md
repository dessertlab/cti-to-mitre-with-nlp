# cti-to-mitre-with-nlp
Replication package for the paper "Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study". 

In this paper, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.

The project includes scripts and data to repeat the training/testing experiments on classification models presented in the paper.

# Project Organization

The diagram below provides the organization of the project:

```
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- bulid_dataset
|   -- cti
|      -- capec
|         -- 2.0
|   -- enterprise_attack
|      -- enterprise-attack-10.1.json
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
|   -- conv_clf.py
|   -- document_analysis.py
|   -- document_data.py
|   -- LSTM_clf.py
|   -- ml_classifier.py
|   -- pretrained_embedding_LSTM.py
|   -- utils
|   -- FIN6
|   -- MenuPass
|   -- WizardSpider
```
In the `build_dataset` directory you can find the script to automatically generate the dataset from MITRE ATT&CK and CAPEC Knowledge bases, distributed in the STIX format. Notice that we loaded into the repo the KB used in our work, but you can pontentially download a newer version from official repositories (link) and update the dataset. 

In the `data` directory you can find the dataset.csv file. It contains the dataset used for the training and evaluation of our classification models. In the `other_datasets` sub-directory there are additional datasets used for validate our preliminary results. 

Finally, in the `src` dataset you can find all the scripts: the ones used for the training/testing of classification models and the ones employed for the document analysis. In the `src/FIN6`, `src/WizardSpider`, and `src/MenuPass` sub-directories you can find the resources chosen and employed in the document analysis. 

# Installation steps and troubleshooting 
- In order to upload bigger files, such as mitre KB, https://git-lfs.github.com was used, so you have to install the LFS extension:
	mac: brew install git-lfs
	
	Download and install the Git command line extension. Once downloaded and installed, set up Git LFS for your user 
	account by running:
	git lfs install
  
  Then you can clone the repo
  
  In case of issues, check out -> https://github.com/git-lfs/git-lfs/wiki/Tutorial#pulling-and-cloning
  
- If you get an error like this:

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

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
