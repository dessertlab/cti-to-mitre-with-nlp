#Plotting results
# importing the required module
import matplotlib.pyplot as plt
import numpy as np

def f_measure(recall, precision):
    if recall != 0 and precision != 0:
        return (2*precision*recall)/(precision+recall)
    else:
        return 0.01

def f_measure_array(recalls, precisions):
    f1_arr = []
    for i in range(0, len(recalls)):
        f1 = f_measure(recalls[i], precisions[i])
        f1_arr.append(f1)
    return f1_arr

def bar_plot(title, thresholds, classifiers_array, classifiers, width = 0.1):
    X_axis = np.arange(len(classifiers))
    for i in range(0, len(thresholds)):
        plt.bar(X_axis+width*i, classifiers_array[i], width, label = thresholds[i], color=thresholds[i])
    
    thick_pos = (len(thresholds)-1)/2
    plt.xticks(X_axis+width*thick_pos, classifiers)

    plt.xlabel('Models')
    plt.ylabel('F_Measure')
    
    #plt.title(title)
    plt.legend(loc='upper right') 
    plt.savefig(title+'.png', format="png", dpi=300)
    plt.close()
    

def read_f1s_from_file(file_name):
    lines = []
    with open(file_name) as f:
        lines = f.readlines()

    classifiers = []
    classifiers_f1 = []
    for line in lines:
        elements = line.split(';')
        classifiers.append(elements[0])
        arr = [float(elem) for elem in elements[1:]]
        classifiers_f1.append(arr)
    return classifiers, classifiers_f1

def prepare_data_for_plotting(thresholds, classifiers_f1):
    classifiers_array = []
    for i in range(0, len(thresholds)):
        arr = []
        for classifier in classifiers_f1:
            arr.append(classifier[i])
        classifiers_array.append(arr)
    return classifiers_array

#classifiers = ['NB', 'MLP', 'CNN', 'OVR', 'LR', 'LSTM', 'pre_LSTM']
def main():
    thresholds = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    documents = ['FIN6/FIN6_ref_1', 'FIN6/FIN6_ref_2', 'MenuPass/MenuPass_ref_8', 'MenuPass/MenuPass_ref_2',
                    'WizardSpider/WizardSpider_ref_7', 'WizardSpider/WizardSpider_ref_2']
    #documents = ['FIN6/FIN6_ref_1']
    for doc in documents:
        classifiers, classifiers_f1 = read_f1s_from_file(doc+'_f1.txt')
        classifiers_array = prepare_data_for_plotting(thresholds, classifiers_f1)
        bar_plot(doc+'_secBert', thresholds, classifiers_array, classifiers)


if __name__ == "__main__":
    main()













#---------------------------
# MLP_recalls = [14/17,12/17,10/17,8/17,8/17,7/17,7/17,6/17]
# MLP_precisions = [14/27,12/19,10/15,8/12,8/11,7/9,7/7,6/6]

# logreg_Recall = [4/17,1/17,0/17,0/17,0/17,0/17,0/17,0/17]
# logreg_precision = [4/6,1/2,0,0,0,0,0,0]

# #Multinomial_NB
# nb_Recall = [5/17,3/17,2/17,1/17,1/17,1/17,1/17,0/17]
# nb_precision = [5/7,3/3,2/2,1/1,1/1,1/1,1/1,0]

# #SVM_Classifier_OVO
# ovo_Recall = [11/17,9/17,6/17,5/17,3/17,1/17,1/17,1/17]
# ovo_precision = [11/19,9/13,6/10,5/6,3/4,1/2,1/2,1/1]

# #SVM_Classifier_OVR
# ovr_Recall = [11/17,10/17,7/17,6/17,4/17,1/17,1/17,1/17]
# ovr_precision = [11/18,10/14,7/11,6/8,4/5,1/2,1/2,1/1]

# #Logreg_normale
# logregnorm_Recall = [8/17,4/17,4/17,4/17,2/17,0/17,0/17,0/17]
# logregnorm_precision = [8/13,4/6,4/6,4/5,2/2,0,0,0]

# #CNN
# cnn_recall = [11/17,10/17,10/17,10/17,10/17,8/17,6/17,4/17]
# cnn_precision = [11/43,10/39,10/30,10/22,10/18,8/10,6/8,4/6]

# #LSTM
# lstm_recall = [11/17,11/17,10/17,10/17,8/17,7/17,6/17,6/17]
# lstm_precision = [11/49,11/45,10/34,10/28,8/21,7/15,6/11,6/8]

# #pretrained_LSTM
# pre_lstm_recall = [8/17,8/17,8/17,7/17,4/17,4/17,3/17,3/17]
# pre_lstm_precision = [8/38,8/35,8/25,7/16,4/10,4/7,3/5,3/5]


# mlp_f1 = f_measure_array(MLP_recalls, MLP_precisions)
# logreg_f1 = f_measure_array(logreg_Recall, logreg_precision)
# ovo_f1 = f_measure_array(ovo_Recall, ovo_precision)
# ovr_f1 = f_measure_array(ovr_Recall, ovr_precision)
# logregnorm_f1 = f_measure_array(logregnorm_Recall, logregnorm_precision)
# cnn_f1 = f_measure_array(cnn_recall, cnn_precision)
# lstm_f1 = f_measure_array(lstm_recall, lstm_precision)
# pre_lstm_f1 = f_measure_array(pre_lstm_recall, pre_lstm_precision)
# nb_f1 = f_measure_array(nb_Recall, nb_precision)

# classifiers_f1 = [nb_f1, mlp_f1, cnn_f1, ovr_f1, logreg_f1, lstm_f1, pre_lstm_f1]