import pandas as pd
from sklearn.model_selection import train_test_split

# #KERAS 
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from deepl_utils import *

import os 

TRAINING_SIZE = 0.80

def main():

    config = LSTM_model_config()

    pp_manager = DLPreprocessingManager()

    data_df = pd.read_csv('../data/dataset.csv')

    num_classes = len(data_df['label_tec'].value_counts())
    print(num_classes)

    path = './lstm_model'
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)  

    pp_manager.fit(sentences=data_df['sentence'], labels=data_df['label_tec'])
    pp_manager.save_preprocessing_pipe(path=path)

    Y = pp_manager.get_labels_encoding(data_df['label_tec'])

    X_train, X_test, Y_train, Y_test = train_test_split(data_df['sentence'].values,data_df['label_tec'], test_size = 0.20, random_state = 42, stratify=data_df['label_tec'])

    #Transform sentences - TRAIN
    X_train_vec = pp_manager.get_features_vectors(X_train, config.MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X_train_vec.shape)
    #Transform label 
    Y_train_vec = pp_manager.get_labels_encoding(Y_train)

    #Transform sentences - TEST
    X_test_vec = pp_manager.get_features_vectors(X_test, config.MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X_test_vec.shape)
    #Transform label 
    Y_test_vec = pp_manager.get_labels_encoding(Y_test)

    #------------LSTM----------------
    epochs = 100
    batch_size = 64
    nn_model = KerasClassifier(model=MODELS.LSTM, num_outputs=num_classes, MAX_NB_WORDS = config.MAX_NB_WORDS, EMBEDDING_DIM = config.EMBEDDING_DIM, 
                                MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH, epochs=epochs, batch_size=batch_size, validation_split=0.2, 
                                loss="categorical_crossentropy", callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    nn_model.fit(X_train_vec,Y_train_vec)

    model_manager = Model_Manager(nn_model)
    model_manager.save_model(path=path)
    precision, recall, fscore, topk = model_manager.calculate_metrics(sentences_vec=X_test_vec, labels_vec=Y_test_vec, labels=Y)

    print("[LSTM]Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore) + " AC@3: " + str(topk) + "\n")


if __name__ == "__main__":
    main()
