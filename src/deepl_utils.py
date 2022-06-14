# #KERAS 
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

from scikeras.wrappers import KerasClassifier

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from scikeras.wrappers import KerasClassifier

from keras.layers import Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, Embedding

import pickle
import enum

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

    def get_tokenizer_vocab(self,):
        return self.tokenizer.word_index

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
    
    def __init__(self, model = KerasClassifier()):
        self.model = model
    
    def save_model(self, path):
        filename = path+'/saved_model' + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load_model(self, path):
        filename = path+'/saved_model' + '.sav'
        self.model = pickle.load(open(filename, 'rb'))
        return self.model

    def calculate_metrics(self,sentences_vec, labels_vec, labels):
        Y_pred = self.model.predict(sentences_vec)

        predict_proba_scores = self.model.predict_proba(sentences_vec)

        sample_weights = compute_sample_weight(class_weight='balanced', y=labels_vec)

        uni_labels = unique_labels(labels)
        print(sample_weights.shape)

        precision, recall, fscore, support = precision_recall_fscore_support(labels_vec, Y_pred, average='weighted')
        topk = top_k_accuracy_score(labels_vec, predict_proba_scores, k=3, labels=uni_labels, sample_weight=sample_weights)

        return precision, recall, fscore, topk

def cnn_model(num_outputs, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):

    EMBEDDING_DIM = 100
    MAX_NB_WORDS = 50000

    embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(256,5,activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print(model.summary())
    return model 

def lstm_model(num_outputs, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    nn_model = Sequential()
    nn_model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    nn_model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    nn_model.add(Dense(num_outputs, activation='softmax'))
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nn_model.summary())
    return nn_model

def pretrained_lstm_model(num_outputs, vocab, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    nn_model = Sequential()
    nn_model.add(Embedding(input_dim=len(vocab)+1, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    nn_model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))
    nn_model.add(Dense(num_outputs, activation='softmax'))
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nn_model.summary())
    return nn_model

class MODELS(enum.Enum):
    CNN = cnn_model
    LSTM = lstm_model
    PRETRAINED_LSTM = pretrained_lstm_model

class CNN_model_config:

    def __init__(self, EMBEDDING_DIM = 100, MAX_NB_WORDS = 50000, MAX_SEQUENCE_LENGTH =50):
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def get_saving_path(self,):
        return 'cnn_model'
        

class LSTM_model_config:

    def __init__(self, EMBEDDING_DIM = 300, MAX_NB_WORDS = 50000, MAX_SEQUENCE_LENGTH = 20):
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def get_saving_path(self,):
        return 'lstm_model'

class LSTM_pretrained_config:

    def __init__(self, MAX_NB_WORDS = 10000, MAX_SEQUENCE_LENGTH = 20, EMBEDDING_DIM = 100):
        self.MAX_NB_WORDS = MAX_NB_WORDS
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def get_saving_path(self,):
        return 'pretrained-lstm_model'
