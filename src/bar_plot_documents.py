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
