
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split

# #KERAS 
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from scikeras.wrappers import KerasClassifier

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

from nltk.tokenize import sent_tokenize
from utils.csv_output import Classifier_results, CSVOutput
import re
from utils.filter_data import *

# import ..utils.csv_output as csv
# import ..utils.filter_data as filtered


TRAINING_SIZE = 0.80

data_df = pd.read_csv('../data/dataset.csv')

data_df['sentence'] = data_df['sentence'].astype(str)

num_classes = len(data_df['label_tec'].value_counts())

print(num_classes)

model = Word2Vec.load("./model/1million.word2vec.model")

#Expand the model with new terms of our corpus - Incremental training
stop_words = set(stopwords.words('english'))
x_tokenized = [[w for w in sentence.split(" ") if w != " " and not w.lower() in stop_words] for sentence in data_df.sentence]
print("Before vocab: " + str(len(model.wv)) + "\n")
model.build_vocab(x_tokenized, update=True)
print("After vocab: " + str(len(model.wv)) + "\n")
# train existing model on new terms
model.train(x_tokenized, total_examples=model.corpus_count, epochs=model.epochs)

 
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 10000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 20
# This is fixed.
EMBEDDING_DIM = 100

#Prepare the input for the model  
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-.:;<=>?@[\]^_`{|}~\n', lower=True)#/ removed from filters in order to preserve pathnames
tokenizer.fit_on_texts(data_df['sentence'].values)
vocab = tokenizer.word_index
print('Found %s unique tokens.' % len(vocab))
X = tokenizer.texts_to_sequences(data_df['sentence'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#Encoding label values in one-hot vectors
encoder = LabelEncoder()
encoder.fit(data_df['label_tec'])
Y = encoder.transform(data_df['label_tec'])
 
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = (1-TRAINING_SIZE), random_state = 42, stratify=Y)
print(X_train.shape,Y_train.shape)

#Prepare the emebedding matrix for initializing the embedding layer with a pre-trained word2vec
#Each element of the vocabulary will be associated with the related pre-trained vector
embedding_matrix = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
for word, i in vocab.items():
    if word in model.wv:
        embedding_vector = model.wv[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

def create_model():
    nn_model = Sequential()
    nn_model.add(Embedding(input_dim=len(vocab)+1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    nn_model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    nn_model.add(Dense(num_classes, activation='softmax'))
    nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nn_model.summary())
    return nn_model

epochs = 100
batch_size = 32

nn_model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

#Train
nn_model.fit(X_train,Y_train)

#Test
Y_pred = nn_model.predict(X_test)
predict_proba_scores = nn_model.predict_proba(X_test)
sample_weights = compute_sample_weight(class_weight='balanced', y=Y_test)
labels = unique_labels(Y)

precision, recall, fscore, support = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
topk = top_k_accuracy_score(Y_test, predict_proba_scores, k=3, labels=labels, sample_weight=sample_weights)

# get the metrics
print("Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore) + " AC@3: " + str(topk) + "\n")

#-------------------------------------------DOCUMENT ANALYSIS----------------------------------------------

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

def f_measure(recall, precision):
    if recall != 0 and precision != 0:
        return (2*precision*recall)/(precision+recall)
    else:
        return 0.01

def analyze_all_doc(file_path, tecs_vec):

    lines = []


    with open(file_path) as f:
        lines += f.readlines()

    ## Apply regex 
    regex_list = load_regex("regex.yml")

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

    predictions = nn_model.predict(X)
    predicted = encoder.inverse_transform(predictions)

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
            top_classes = encoder.inverse_transform(top_class_v[i])
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

    result = Classifier_results( title='pretrained_LSTM', 
                                lines=num_sen,
                                accepted_preds=accepted_pred, 
                                correct_preds=corrected_pred, 
                                precisions=precisions, 
                                recalls=recalls, 
                                correct_uniques=correct_on_uniques,
                                f1s=f1s)
    return result

# FIN6

fin6_tecs_intel = ['T1078', 'T1133', 'T1566', ' T1046', 'T1069', 'T1068', 'T1018', 'T1105', 'T1553', 'T1547', 'T1053', 'T1021', 
            'T1059', 'T1047', 'T1003', 'T1048', 'T1005', 'T1027', 'T1560', 'T1074', 'T1190', 'T1072', 'T1080', 'T1484']

fin6_tec_2 = ['T1134', 'T1059', 'T1562', 'T1036', 'T1588', 'T1003', 'T1021', 'T1569', 'T1078', 'T1102',
        'T1087', 'T1482', 'T1069', 'T1018', 'T1016', 'T1548', 'T1071', 'T1185', 'T1059', 
        'T1543', 'T1132', 'T1005', 'T1001', 'T1140', 'T1573', 'T1068', 'T1083',
        'T1564', 'T1562', 'T1070', 'T1105', 'T1056', 'T1112', 'T1046', 'T1095', 'T1027', 'T1137', 
        'T1003', 'T1069', 'T1057', 'T1055', 'T1572', 'T1572', 'T1090', 'T1012', 'T1620', 'T1021',
        'T1018', 'T1029', 'T1113', 'T1518', 'T1553', 'T1218', 'T1049', 'T1007', 'T1569', 'T1550', 
        'T1078', 'T1047']

fin6_tec_1 = ['T1087', 'T1560', 'T1119', 'T1547', 'T1110', 'T1059', 'T1074', 'T1573', 'T1068', 'T1070', 
        'T1046', 'T1003', 'T1572', 'T1021', 'T1018', 'T1053', 'T1078', 'T1003']

# MenuPass [8]

menuPass_tec_8 = ['T1560', 'T1119', 'T1059', 'T1005', 'T1074', 'T1210', 'T1083', 'T1574', 'T1106', 'T1027', 'T1003', 'T1199', 'T1078', 'T1047']

adFind = ['T1087', 'T1482', 'T1069', 'T1018', 'T1016']

certutil = ['T1140', 'T1105', 'T1553']

quasarRAT = ['T1059', 'T1555', 'T1573', 'T1105', 'T1056', 'T1112', 'T1090', 'T1021', 'T1053', 'T1553', 'T1082', 'T1552', 'T1125']

menuPass_tec_8.extend(adFind)
menuPass_tec_8.extend(certutil)
menuPass_tec_8.extend(quasarRAT)

# MenuPass [2]

menuPass_tec_2 = ['T1583', 'T1560', 'T1568', 'T1070', 'T1056', 'T1036', 'T1105', 'T1566', 'T1021', 'T1199', 'T1204', 'T1078']

poisonIvy = ['T1010', 'T1547', 'T1059', 'T1543', 'T1005', 'T1074', 'T1573', 'T1105', 'T1056', 'T1112', 'T1027', 
'T1055', 'T1014']

menuPass_tec_2.extend(poisonIvy)

# WizardSpider [2]

wizardSpider_tec_2 = ['T1547', 'T1059', 'T1562', 'T1135', 'T1566', 'T1055', 'T1021', 'T1053', 'T1558', 'T1204', 'T1047']

bloodHound = ['T1087', 'T1560', 'T1059', 'T1482', 'T1615', 'T1106', 'T1201', 'T1069', 'T1018', 'T1033']

cobaltStrike = ['T1548', 'T1134', 'T1087', 'T1071', 'T1197', 'T1185', 'T1059', 'T1043', 'T1543', 'T1132', 'T1005', 'T1001', 'T1030', 'T1140', 'T1573', 'T1203', 'T1068', 'T1083', 'T1564', 'T1562', 'T1070', 'T1105', 'T1056', 'T1112', 'T1026', 'T1106', 'T1046', 'T1135', 'T1095', 'T1027', 'T1137', 'T1003', 'T1069', 'T1057', 'T1055', 'T1572', 'T1090', 'T1012', 'T1620', 'T1021', 'T1018', 'T1029', 'T1113', 'T1518', 'T1553', 'T1218', 'T1016', 'T1049', 'T1007', 'T1569', 'T1550', 'T1078', 'T1047']

empire =  ['T1548', 'T1134', 'T1087', 'T1557', 'T1071', 'T1560', 'T1547', 'T1217', 'T1115', 'T1059', 'T1043', 'T1136', 'T1543', 'T1555', 'T1484', 'T1482', 'T1114', 'T1573', 'T1546', 'T1068', 'T1083', 'T1574', 'T1210', 'T1615', 'T1567', 'T1070',  'T1056', 'T1105', 'T1056', 'T1106', 'T1046', 'T1135', 'T1040', 'T1027', 'T1003', 'T1057', 'T1055', 'T1021', 'T1053', 'T1113', 'T1518', 'T1558', 'T1082', 'T1016', 'T1049', 'T1569', 'T1127', 'T1552', 'T1550', 'T1125', 'T1102', 'T1047']

mimikatz = ['T1134', 'T1098', 'T1547', 'T1555', 'T1003', 'T1207', 'T1558', 'T1552', 'T1550']

ping = ['T1018']

ryuk = ['T1134', 'T1547', 'T1059', 'T1486', 'T1083', 'T1222', 'T1562', 'T1490', 'T0828', 'T1036', 'T1106', 'T1027', 'T1057', 'T1055', 'T1021', 'T1053', 'T1489', 'T1082', 'T1614', 'T1016', 'T1205', 'T1078']

trickBot = ['T1087', 'T1087', 'T1071', 'T1547', 'T1185', 'T1110', 'T1059', 'T1059', 'T1043', 'T1543', 'T1555', 'T1555', 'T1132', 'T1005', 'T1140', 'T1482', 'T1573', 'T1041', 'T1210', 'T1008', 'T1083', 'T1495', 'T1562', 'T1105', 'T1056', 'T1559', 'T1036', 'T1112', 'T1106', 'T1135', 'T1571', 'T1027', 'T1027', 'T1069', 'T1566', 'T1566', 'T1542', 'T1057', 'T1055', 'T1055', 'T1090', 'T1219', 'T1021', 'T1018', 'T1053', 'T1553', 'T1082', 'T1016', 'T1033', 'T1007', 'T1552', 'T1552', 'T1204', 'T1497']

wizardSpider_tec_2.extend(bloodHound)
wizardSpider_tec_2.extend(cobaltStrike)
wizardSpider_tec_2.extend(empire)
wizardSpider_tec_2.extend(mimikatz)
wizardSpider_tec_2.extend(ping)
wizardSpider_tec_2.extend(ryuk)
wizardSpider_tec_2.extend(trickBot)

#WizardSpider [7]

wizardSpider_tec_7 = ['T1087', 'T1059', 'T1048', 'T1210', 'T1562', 'T1027', 'T1021', 'T1018', 'T1489', 'T1518', 'T1558', 'T1082', 'T1569']

adFind = ['T1087', 'T1482', 't1069', 'T1018', 'T1016']

#CobaltStrike

net = ['T1087', 'T1087', 'T1136', 'T1136', 'T1070', 'T1135', 'T1201', 'T1069', 'T1069', 'T1021', 'T1018', 'T1049', 'T1007', 'T1569', 'T1124']

nltest = ['T1482', 'T1018', 'T1016']

#Ping

#Ryuk

wizardSpider_tec_7.extend(adFind)
wizardSpider_tec_7.extend(cobaltStrike)
wizardSpider_tec_7.extend(net)
wizardSpider_tec_7.extend(nltest)
wizardSpider_tec_7.extend(ping)
wizardSpider_tec_7.extend(ryuk)


# ------------------------------------

fin6_files = ['FIN6/Follow The Money-Dissecting the Operations of the Cyber Crime Group FIN6[1].txt', 
                'FIN6/Pick-Six-Intercepting a FIN6 Intrusion, an Actor Recently Tied to Ryuk and LockerGoga Ransomware[2].txt',
                'FIN6/intelligence_summary.txt']

menuPass_files = ['MenuPass/2018_12_20_united_states_v_zhu_hua_indictment[2].txt', 
                'MenuPass/Japan-Linked Organizations Targeted in Long-Running and Sophisticated Attack Campaign[8].txt']

wizardSpider_files = ['WizardSpider/Ryuk’s Return[7].txt',
                     'WizardSpider/Ransomware Activity Targeting the Healthcare and Public Health Sector. Retrieved October 28, 2020[2].txt']

# ------------------------------------

# fin6_intel_results = analyze_all_doc(fin6_files[2], 
#                             fin6_tecs_intel)
# fin6_intel_output = CSVOutput('FIN6/FIN6_intelligence_summary', [fin6_intel_results])
# fin6_intel_output.append_to_file('.')

# fin6_ref_1_results = analyze_all_doc(fin6_files[0], 
#                             fin6_tec_1)
# fin6_ref_1_output = CSVOutput('FIN6/FIN6_ref_1', [fin6_ref_1_results])
# fin6_ref_1_output.append_to_file('.')

# fin6_ref_2_results = analyze_all_doc(fin6_files[1], 
#                             fin6_tec_2)
# fin6_ref_2_output = CSVOutput('FIN6/FIN6_ref_2', [fin6_ref_2_results])
# fin6_ref_2_output.append_to_file('.')

# menuPass_ref_8_results = analyze_all_doc(menuPass_files[1], 
#                             menuPass_tec_8)
# menuPass_ref_8_output = CSVOutput('MenuPass/MenuPass_ref_8', [menuPass_ref_8_results])
# menuPass_ref_8_output.append_to_file('.')

# menuPass_ref_2_results = analyze_all_doc(menuPass_files[0], 
#                             menuPass_tec_2)
# menuPass_ref_2_output = CSVOutput('MenuPass/MenuPass_ref_2', [menuPass_ref_2_results])
# menuPass_ref_2_output.append_to_file('.')

# wizardSpider_ref_7_results = analyze_all_doc(wizardSpider_files[0], 
#                             wizardSpider_tec_7)
# wizardSpider_ref_7_output = CSVOutput('WizardSpider/WizardSpider_ref_7', [wizardSpider_ref_7_results])
# wizardSpider_ref_7_output.append_to_file('.')

# wizardSpider_ref_2_results = analyze_all_doc(wizardSpider_files[1], 
#                             wizardSpider_tec_2)
# wizardSpider_ref_2_output = CSVOutput('WizardSpider/WizardSpider_ref_2', [wizardSpider_ref_2_results])
# wizardSpider_ref_2_output.append_to_file('.')
