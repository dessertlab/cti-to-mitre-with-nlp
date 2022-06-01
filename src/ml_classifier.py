from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer, porter
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier 
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

from utils.filter_data import *

def lemmatize_set(dataset):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for sentence in dataset:
        word_list = word_tokenize(sentence)
        lemma_list = [lemmatizer.lemmatize(w) for w in word_list]
        lemmatized_list.append(' '.join(lemma_list))
    return lemmatized_list

def stemmatize_set(dataset):
    ps = porter.PorterStemmer()
    stemmatize_list = []
    for sentence in dataset:
        word_list = word_tokenize(sentence)
        stemma_list = [ps.stem(w) for w in word_list]
        stemmatize_list.append(' '.join(stemma_list))
    return stemmatize_list

def print_k_likely_results(prob_v, sorted_index_v, class_name_v, k):
    print("Top k classes are: \n")
    for i in range(0,k):
        inv = k - 1 - i
        print( str(i+1) + "candidate is: " + class_name_v[inv] + " with a probability of " + str(prob_v[sorted_index_v[inv]]) + "\n")

TRAINING_SIZE = 0.80

data_df = pd.read_csv('../data/dataset.csv')
num_classes = len(data_df['label_tec'].value_counts())

print(num_classes)

data_df['sentence'] = data_df['sentence'].astype(str)

#Cleaning sentences 
vectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english', max_features=10000, ngram_range=(1,2))

stemmatized_set = stemmatize_set(data_df.sentence)
lemmatized_set = lemmatize_set(stemmatized_set)
x_train_vectors = vectorizer.fit_transform(lemmatized_set)

bow_vocab = vectorizer.get_feature_names_out()

mnb_clf = MultinomialNB()
cnb_clf = ComplementNB()
logisticRegr = LogisticRegression(class_weight='balanced', multi_class='multinomial')
logisticRegr_norm = LogisticRegression()
knn_clf=KNeighborsClassifier()
clf_rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
nn_clf = MLPClassifier(max_iter=1000, early_stopping=True)



#Train the SVM model
clf_svm = svm.SVC(kernel='linear', probability=True, class_weight='balanced', decision_function_shape='ovo')
clf_svm_linear = svm.SVC(kernel='linear', probability=True, class_weight='balanced', decision_function_shape='ovr')
encoder = LabelBinarizer()
encoder.fit(data_df.label_tec)
y_enc = encoder.transform(data_df.label_tec)
text_label = encoder.classes_

## Helper function for regex: creates a label based on a specific windows path 
def repl(matchobj):
    return matchobj.group(2) + "_path" #the path name string is captured by group 2


def train_classifier(classifier, name, X, Y):
    #train sets and test sets
    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, 
                                                        test_size=(1-TRAINING_SIZE),  random_state=4, stratify=Y)

    
    stemmatized_set = stemmatize_set(train_set_x)
    lemmatized_set = lemmatize_set(stemmatized_set)
    x_train_vectors = vectorizer.fit_transform(lemmatized_set)
    print(x_train_vectors.shape)
    classifier.fit(x_train_vectors, train_set_y)

    print("Model has been trained!")
    print(test_set_y.shape)
    stemmatized_set = stemmatize_set(test_set_x)
    lemmatized_set = lemmatize_set(stemmatized_set)
    x_test_vectors = vectorizer.transform(lemmatized_set)
    predicted = classifier.predict(x_test_vectors)
    print(predicted.shape)
    k=3

    ## TODO: de-comment the line for saving 
    # #Saving predictions
    # test_len = len(test_set_x)
    # sentences = []
    # actual_labels = []
    # predicted_labels = []
    # for i in range(test_len):
    #     prediction = classifier.predict(x_test_vectors[i])
    #     predicted_label = prediction[0]
    #     predicted_labels.append(predicted_label)
    #     actual_labels.append(test_set_y.iloc[i])
    #     sentences.append(test_set_x.iloc[i])
    #     # print(test_set_x.iloc[i])
    #     # print('Actual label:' + test_set_y.iloc[i])
    #     # print("Predicted label: " + predicted_label)
    
    # #DataFrame 
    # data = {'sentences': sentences, 'actual_label': actual_labels, 'predicted_label': predicted_labels}
    # pd.set_option('max_colwidth',1000)
    # data_df = pd.DataFrame(data, columns=['sentences', 'actual_label', 'predicted_label'])
    # #Saving on file predictions
    # data_df.to_csv('prediction_mlp.csv')#, index=False)

    #cnf_matrix = confusion_matrix(test_set_y, predicted)

    # plot_confusion_matrix(cnf_matrix, classes=np.asarray(text_labels), normalize=True,
    #                   title='Normalized confusion matrix')

    #Vector of probabilities
    predict_proba_scores = classifier.predict_proba(x_test_vectors)
    #Identify the indexes of the top predictions
    top_k_predictions = np.argsort(predict_proba_scores, axis = 1)[:,-k:]

    #Get classes related to indexes
    top_class = classifier.classes_[top_k_predictions]

    
    labels = unique_labels(Y)
    sample_weights = compute_sample_weight(class_weight='balanced', y=test_set_y)
    print(sample_weights.shape)
    
    precision, recall, fscore, support = precision_recall_fscore_support(test_set_y, predicted, average='weighted')
    topk = top_k_accuracy_score(test_set_y, predict_proba_scores, k=3, labels=labels, sample_weight=sample_weights)

    # get the metrics
    print("Results for" + name + "\n")
    print("Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore) + " AC@3: " + str(topk) + "\n")

    # # Uncomment save the model to disk
    # filename = name + '.sav'
    # pickle.dump((vectorizer, classifier), open(filename, 'wb'))

train_classifier(nn_clf, "MLP classifier ",  data_df.sentence, data_df.label_tec)
train_classifier(logisticRegr, "Logreg",  data_df.sentence, data_df.label_tec)
train_classifier(logisticRegr_norm, "Logreg_normale",  data_df.sentence, data_df.label_tec)
train_classifier(mnb_clf, "Multinomial_NB",  data_df.sentence, data_df.label_tec)
train_classifier(cnb_clf, "Complement_NB",  data_df.sentence, data_df.label_tec)
train_classifier(clf_svm_linear, "SVM_Classifier_OVR", data_df.sentence, data_df.label_tec)
train_classifier(clf_svm, "SVM_Classifier_OVO", data_df.sentence, data_df.label_tec)

