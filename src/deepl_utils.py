# #KERAS 
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

import pickle

class DLPreprocessingManager:

    def __init__(self, MAX_NB_WORDS = 50000, filters='!"#$%&()*+,-.:;<=>?@[\]^_`{|}~\n'):
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=filters, lower=True)
        self.label_encoder = LabelEncoder()

    def fit(self, sentences, labels):
        self.tokenizer.fit_on_texts(sentences.astype(str))
        self.label_encoder.fit(labels)

    def get_features_vectors(self, sentences, MAX_SEQUENCE_LENGTH = 50):
        X = self.tokenizer.texts_to_sequences(sentences)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        return X

    def get_labels_encoding(self, labels):
        return self.label_encoder.transform(labels)

    def get_labels_from_encoding(self, labels):
        return self.label_encoder.inverse_transform(labels)

    def save_preprocessing_pipe(self, path):
        with open(path+'/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path+'/encoder.pickle', 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_preprocessing_pipe(self, path):
        with open(path+'/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        with open(path+'/encoder.pickle', 'rb') as handle:
            self.label_encoder = pickle.load(handle)

class Model_Manager:
    
    def __init__(self, model):
        self.model = model
    
    def save_model(self, path):
        filename = path+'/saved_model' + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load_model(self, path):
        filename = path+'/saved_model' + '.sav'
        self.model = pickle.load(open(filename, 'rb'))

    def calculate_metrics(self,sentences_vec, labels_vec, labels):
        Y_pred = self.model.predict(sentences_vec)

        predict_proba_scores = self.model.predict_proba(sentences_vec)

        sample_weights = compute_sample_weight(class_weight='balanced', y=labels_vec)

        uni_labels = unique_labels(labels)
        print(sample_weights.shape)

        precision, recall, fscore, support = precision_recall_fscore_support(labels_vec, Y_pred, average='weighted')
        topk = top_k_accuracy_score(labels_vec, predict_proba_scores, k=3, labels=uni_labels, sample_weight=sample_weights)

        return precision, recall, fscore, topk


# pp_manager = DLPreprocessingManager()
# pp_manager.load_preprocessing_pipe('cmodel')
# vects = pp_manager.get_features_vectors(['The malware is evil'])
# label = pp_manager.get_labels_encoding(['T1059'])
# print(str(vects)+str(label))