from cProfile import label
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#KERAS 
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from scikeras.wrappers import KerasClassifier

#for Testing
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

#for documents
from nltk.tokenize import sent_tokenize
from utils.csv_output import Classifier_results, CSVOutput
import re
from utils.filter_data import *


def f_measure(recall, precision):
    if recall != 0 and precision != 0:
        return (2*precision*recall)/(precision+recall)
    else:
        return 0.01

TRAINING_SIZE = 0.80

data_df = pd.read_csv('../data/dataset.csv')

data_df['sentence'] = data_df['sentence'].astype(str)

num_classes = len(data_df['label_tec'].value_counts())

print(num_classes)


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 20 
# This is fixed.
EMBEDDING_DIM = 300

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data_df['sentence'].values)
word_index = tokenizer.word_index
len_vocab = len(word_index) + 1

print('Found %s unique tokens.' % len_vocab)

X = tokenizer.texts_to_sequences(data_df['sentence'].values)
stop_words = set(stopwords.words('english'))

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
#Y = pd.get_dummies(data_df['label']).values

#Transform label 
Y = data_df['label_tec']
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#Y = pd.get_dummies(data_df['label_tec']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = (1-TRAINING_SIZE), random_state = 42, stratify=Y)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

epochs = 100
batch_size = 64

def create_model():
    nn_model = Sequential()
    nn_model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    nn_model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    nn_model.add(Dense(num_classes, activation='softmax'))
    nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nn_model.summary())
    return nn_model


nn_model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


nn_model.fit(X_train,Y_train)
Y_pred = nn_model.predict(X_test)

predict_proba_scores = nn_model.predict_proba(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=Y_test)

labels = unique_labels(Y)
print(sample_weights.shape)

precision, recall, fscore, support = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
topk = top_k_accuracy_score(Y_test, predict_proba_scores, k=3, labels=labels, sample_weight=sample_weights)

# get the metrics
print("Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore) + " AC@3: " + str(topk) + "\n")

def remove_empty_lines(text):
	lines = text.split("\n")
	non_empty_lines = [line for line in lines if line.strip() != ""]

	string_without_empty_lines = ""
	for line in non_empty_lines:
		if line != "\n": 
			string_without_empty_lines += line + "\n"

	return string_without_empty_lines 

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

## Helper function for regex: creates a label_tec based on a specific windows path 
def repl(matchobj):
    return matchobj.group(2) + "_path" #the path name string is captured by group 2


def analyze_all_doc(file_path, tecs_vec):

    lines = []


    with open(file_path) as f:
        lines += f.readlines()

    ## Apply regex 
    regex_list = load_regex("utils/regex.yml")

    text = combine_text(lines)
    text = re.sub('(%(\w+)%(\/[^\s]+))', repl, text)
    text = apply_regex_to_string(regex_list, text)
    text = re.sub('\(.*?\)', '', text)
    text = remove_empty_lines(text)
    text = text.strip()
    sentences = sent_tokenize(text)

    double_sentences = []

    for i in range(1, len(sentences)):
        new_sen = sentences[i-1] + sentences[i]
        double_sentences.append(new_sen)

    #Transform sentences
    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    predicted = nn_model.predict(X)

    #Matrix
    #Vector of vector of probabilities
    predict_proba_scores = nn_model.predict_proba(X)
    #Identify the indexes of the top predictions (increasing order so let's take the last 2, highest proba)
    top_k_predictions = np.argsort(predict_proba_scores, axis = 1)[:,-2:]
    #Get classes related to previous indexes
    top_class_v = nn_model.classes_[top_k_predictions]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    precisions = []
    recalls = []
    corrected_pred = []
    accepted_pred = []
    correct_on_uniques = []
    f1s = []

    print(len(predicted))

    num_sen = len(sentences)

    for threshold in thresholds: 
        tecs = tecs_vec
        tecs = set(tecs)    
        accepted = []

        for i in range(0,len(predict_proba_scores)):
            sorted_indexes = top_k_predictions[i]
            top_classes = top_class_v[i]
            proba_vector = predict_proba_scores[i]
            if proba_vector[sorted_indexes[1]] > threshold:
                accepted.append(top_classes[1])

        correct = 0

        unique_accepted = set(accepted)
        len_tecs = len(tecs)

        for pred in accepted:
            if pred in tecs: #True Positives
                correct += 1

        print(correct)

        if len(accepted) != 0:
            precision = correct/len(accepted)*100
        else:
            precision = 0
        
        precision = round(precision,2)
        print(precision) #accuracy or precision?

        precisions.append(precision)

        for pred in accepted:
            if pred in tecs:
                tecs.remove(pred)

        recall = str(len_tecs-len(tecs))+ '/' + str(len_tecs)

        print(recall) #Recall

        recalls.append(recall)
        recall = (len_tecs-len(tecs))/len_tecs

        corrected_pred.append(correct) 
        accepted_pred.append(len(accepted))
        
        cou = str(len_tecs-len(tecs))+ '/' + str(len(unique_accepted))
        correct_on_uniques.append(cou)
        cou = 0 if len(unique_accepted) == 0 else (len_tecs-len(tecs))/len(unique_accepted)

        f1 = f_measure(recall=recall, precision=cou)
        f1 = round(f1,2)
        f1s.append(f1)

        print("Threshold: " + str(threshold) + ": " + str(cou) + " correct on uniques")

    result = Classifier_results( title='LSTM', 
                                lines=num_sen,
                                accepted_preds=accepted_pred, 
                                correct_preds=corrected_pred, 
                                precisions=precisions, 
                                recalls=recalls, 
                                correct_uniques=correct_on_uniques,
                                f1s=f1s)
    return result

from document_data import *

fin6_intel_results = analyze_all_doc(fin6_files[2], 
                            fin6_tecs_intel)
fin6_intel_output = CSVOutput('FIN6/FIN6_intelligence_summary', [fin6_intel_results])
fin6_intel_output.append_to_file('.')

fin6_ref_1_results = analyze_all_doc(fin6_files[0], 
                            fin6_tec_1)
fin6_ref_1_output = CSVOutput('FIN6/FIN6_ref_1', [fin6_ref_1_results])
fin6_ref_1_output.append_to_file('.')

fin6_ref_2_results = analyze_all_doc(fin6_files[1], 
                            fin6_tec_2)
fin6_ref_2_output = CSVOutput('FIN6/FIN6_ref_2', [fin6_ref_2_results])
fin6_ref_2_output.append_to_file('.')

menuPass_ref_8_results = analyze_all_doc(menuPass_files[1], 
                            menuPass_tec_8)
menuPass_ref_8_output = CSVOutput('MenuPass/MenuPass_ref_8', [menuPass_ref_8_results])
menuPass_ref_8_output.append_to_file('.')

menuPass_ref_2_results = analyze_all_doc(menuPass_files[0], 
                            menuPass_tec_2)
menuPass_ref_2_output = CSVOutput('MenuPass/MenuPass_ref_2', [menuPass_ref_2_results])
menuPass_ref_2_output.append_to_file('.')

wizardSpider_ref_7_results = analyze_all_doc(wizardSpider_files[0], 
                            wizardSpider_tec_7)
wizardSpider_ref_7_output = CSVOutput('WizardSpider/WizardSpider_ref_7', [wizardSpider_ref_7_results])
wizardSpider_ref_7_output.append_to_file('.')

wizardSpider_ref_2_results = analyze_all_doc(wizardSpider_files[1], 
                            wizardSpider_tec_2)
wizardSpider_ref_2_output = CSVOutput('WizardSpider/WizardSpider_ref_2', [wizardSpider_ref_2_results])
wizardSpider_ref_2_output.append_to_file('.')
