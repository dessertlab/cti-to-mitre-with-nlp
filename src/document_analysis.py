from cmath import pi
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, porter
import numpy as np
from utils.filter_data import *
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from utils.csv_output import Classifier_results, CSVOutput
import os

from deepl_utils import *


def repl(matchobj):
    return ","+ matchobj.group(1) + ","

# TODO: Check the func cleaning before using it in this code
def cleaning_data(text): 
	#text = text.lower()							
	text = re.sub('\(i.e.', '', text)
	text = re.sub('\[(.*?)\]', repl, text)
	text = re.sub('\(.*?\)', '', text)
	text = re.sub('\)', '', text)
	text = re.sub('\<\/?code\>', '', text)
	text = text.strip()
	return text

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

def f_measure(recall, precision):
    if recall != 0 and precision != 0:
        return (2*precision*recall)/(precision+recall)
    else:
        return 0.01

def print_k_likely_results(prob_v, sorted_index_v, class_name_v, k):
    print("Top k classes are: \n")
    for i in range(0,k):
        inv = k - 1 - i
        print(str(i+1) + "candidate is: " + class_name_v[inv] + " with a probability of " + str(prob_v[sorted_index_v[inv]]) + "\n")


ml_model_filenames = ['ml_models/MLP_classifier.sav', 'ml_models/Logreg.sav', 'ml_models/Multinomial_NB.sav', 'ml_models/SVM_Classifier_OVR.sav']
                    #, 'SVM_Classifier_OVO.sav', 'Logreg_normale.sav']

def analyze_all_doc(file_path, model_filenames, tecs_vec):
    lines = [] 

    with open(file_path, encoding="utf8", errors='ignore') as f:
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

    print(len(sentences))

    num_sen = len(sentences)

    double_sentences = []

    for i in range(1, len(sentences)):
        new_sen = sentences[i-1] + sentences[i]
        double_sentences.append(new_sen)

    results = []
    
    for model_filename in model_filenames: 
        # load the model from disk
        vectorizer, classifier = pickle.load(open(model_filename, 'rb'))

        stemmatized_set = stemmatize_set(sentences)
        lemmatized_set = lemmatize_set(stemmatized_set)
        x_test_vectors = vectorizer.transform(lemmatized_set)
        
        #Matrix
        #Vector of vector of probabilities
        predict_proba_scores = classifier.predict_proba(x_test_vectors)
        #Identify the indexes of the top predictions (increasing order so let's take the last 2, highest proba)
        top_k_predictions = np.argsort(predict_proba_scores, axis = 1)[:,-2:]
        #Get classes related to previous indexes
        top_class_v = classifier.classes_[top_k_predictions]

        thresholds = [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        precisions = []
        recalls = []
        corrected_pred = []
        accepted_pred = []
        correct_on_uniques = []
        f1s = []

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

        # precisions = []
        # recalls = []
        # corrected_pred = []
        # accepted_pred = []
        # correct_on_uniques = []

        title = os.path.splitext(model_filename)[0]
        title = title.split('/')[1]
        result = Classifier_results( title=title, 
                                        lines=num_sen,
                                        accepted_preds=accepted_pred, 
                                        correct_preds=corrected_pred, 
                                        precisions=precisions, 
                                        recalls=recalls, 
                                        correct_uniques=correct_on_uniques,
                                        f1s=f1s)
        results.append(result)
    
    models = []
    cnn = CNN_model_config()
    lstm = LSTM_model_config()
    models.append(cnn)
    models.append(lstm)
    for model in models:
        pp_manager = DLPreprocessingManager()
        path = model.get_saving_path()
        pp_manager.load_preprocessing_pipe(path=path)
        model_manager = Model_Manager()
        nn_model =model_manager.load_model(path=path)

        #Transform sentences
        X = pp_manager.get_features_vectors(sentences, model.MAX_SEQUENCE_LENGTH)
        predictions = nn_model.predict(X)
        predicted = pp_manager.get_labels_from_encoding(predictions)

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
                top_classes = pp_manager.get_labels_from_encoding(top_class_v[i])
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

        title = path.split('_')[0]
        result = Classifier_results(title=title, 
                                    lines=num_sen,
                                    accepted_preds=accepted_pred, 
                                    correct_preds=corrected_pred, 
                                    precisions=precisions, 
                                    recalls=recalls, 
                                    correct_uniques=correct_on_uniques,
                                    f1s=f1s)
        results.append(result)
    return results



from document_data import *

fin6_intel_results = analyze_all_doc(fin6_files[2], 
                            ml_model_filenames,
                            fin6_tecs_intel)
fin6_intel_output = CSVOutput('FIN6/FIN6_intelligence_summary', fin6_intel_results)
fin6_intel_output.write_to_file('.')

fin6_ref_1_results = analyze_all_doc(fin6_files[0], 
                            ml_model_filenames,
                            fin6_tec_1)
fin6_ref_1_output = CSVOutput('FIN6/FIN6_ref_1', fin6_ref_1_results)
fin6_ref_1_output.write_to_file('.')

fin6_ref_2_results = analyze_all_doc(fin6_files[1], 
                            ml_model_filenames,
                            fin6_tec_2)
fin6_ref_2_output = CSVOutput('FIN6/FIN6_ref_2', fin6_ref_2_results)
fin6_ref_2_output.write_to_file('.')

menuPass_ref_8_results = analyze_all_doc(menuPass_files[1], 
                            ml_model_filenames,
                            menuPass_tec_8)
menuPass_ref_8_output = CSVOutput('MenuPass/MenuPass_ref_8', menuPass_ref_8_results)
menuPass_ref_8_output.write_to_file('.')

menuPass_ref_2_results = analyze_all_doc(menuPass_files[0], 
                            ml_model_filenames,
                            menuPass_tec_2)
menuPass_ref_2_output = CSVOutput('MenuPass/MenuPass_ref_2', menuPass_ref_2_results)
menuPass_ref_2_output.write_to_file('.')

wizardSpider_ref_7_results = analyze_all_doc(wizardSpider_files[0], 
                            ml_model_filenames,
                            wizardSpider_tec_7)
wizardSpider_ref_7_output = CSVOutput('WizardSpider/WizardSpider_ref_7', wizardSpider_ref_7_results)
wizardSpider_ref_7_output.write_to_file('.')

wizardSpider_ref_2_results = analyze_all_doc(wizardSpider_files[1], 
                            ml_model_filenames,
                            wizardSpider_tec_2)
wizardSpider_ref_2_output = CSVOutput('WizardSpider/WizardSpider_ref_2_prova', wizardSpider_ref_2_results)
wizardSpider_ref_2_output.write_to_file('.')























