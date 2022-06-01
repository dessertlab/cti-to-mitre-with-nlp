class Classifier_results:
    def __init__(self, title, lines, accepted_preds, correct_preds, precisions, recalls, correct_uniques, f1s):
        self.title = title
        self.lines = lines
        self.accepted_preds = accepted_preds
        self.correct_preds = correct_preds
        self.precisions = precisions
        self.recalls = recalls
        self.correct_uniques = correct_uniques
        self.f1s = f1s

class CSVOutput:
    def __init__(self, document_title, classifiers):
        self.classifiers = classifiers
        self.document_title = document_title

    def printify_array(self, array, sep = ';'):
        return sep + sep.join(str(x) for x in array)

    def _save_classifier_outputs(self, f):
        for classifier in self.classifiers:
            f.write(classifier.title + '\n')
            f.write(str(classifier.lines) + ' sentences\n')
            f.write('Accepted Predictions: {}\n'.format(self.printify_array(classifier.accepted_preds)))
            f.write('Corrected Predictions: {}\n'.format(self.printify_array(classifier.correct_preds)))
            f.write('Precision%: {}\n'.format(self.printify_array(classifier.precisions)))
            f.write('Recall%: {}\n'.format(self.printify_array(classifier.recalls)))
            f.write('Correct predictions on uniques: {}\n\n'.format(self.printify_array(classifier.correct_uniques)))

    def _save_classifier_f1(self, path):
        with open(path+'/'+self.document_title+'_f1.txt', 'a') as f:
            for classifier in self.classifiers:
                f.write(classifier.title)
                f.write(self.printify_array(classifier.f1s)+ '\n')

    def write_to_file(self, path):
        self._save_classifier_f1(path)
        with open(path+'/'+self.document_title+'.csv', 'w') as f:
            f.write('Tresholds; 0,1; 0,15; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8;\n')
            self._save_classifier_outputs(f)
            
    def append_to_file(self, path):
        self._save_classifier_f1(path)
        with open(path+'/'+self.document_title+'.csv', 'a') as f:
            self._save_classifier_outputs(f)
