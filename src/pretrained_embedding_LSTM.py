import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# #KERAS 
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from gensim.models import Word2Vec
from nltk.corpus import stopwords

from deepl_utils import *

import os 

TRAINING_SIZE = 0.80

def main():

    config = LSTM_pretrained_config()

    pp_manager = DLPreprocessingManager()

    data_df = pd.read_csv('../data/dataset.csv')

    num_classes = len(data_df['label_tec'].value_counts())
    print(num_classes)

    path = './'+config.get_saving_path()
    try:
        os.mkdir(path)
    except OSError as error:
        print(error) 

    model = Word2Vec.load("../model/1million.word2vec.model")

    #Expand the model with new terms of our corpus - Incremental training
    stop_words = set(stopwords.words('english'))
    x_tokenized = [[w for w in sentence.split(" ") if w != " " and not w.lower() in stop_words] for sentence in data_df.sentence]
    print("Before vocab: " + str(len(model.wv)) + "\n")
    model.build_vocab(x_tokenized, update=True)
    print("After vocab: " + str(len(model.wv)) + "\n")
    # train existing model on new terms
    model.train(x_tokenized, total_examples=model.corpus_count, epochs=model.epochs)

    pp_manager.fit(sentences=data_df['sentence'], labels=data_df['label_tec'])
    vocab = pp_manager.get_tokenizer_vocab()
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

    #Prepare the emebedding matrix for initializing the embedding layer with a pre-trained word2vec
    #Each element of the vocabulary will be associated with the related pre-trained vector
    embedding_matrix = np.zeros((len(vocab) + 1, config.EMBEDDING_DIM))
    for word, i in vocab.items():
        if word in model.wv:
            embedding_vector = model.wv[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    #------------pretrained-LSTM----------------
    epochs = 100
    batch_size = 32
    nn_model = KerasClassifier(model=MODELS.PRETRAINED_LSTM, num_outputs=num_classes, vocab=vocab, embedding_matrix=embedding_matrix, EMBEDDING_DIM = config.EMBEDDING_DIM, 
                                MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH, epochs=epochs, batch_size=batch_size, validation_split=0.2, 
                                loss="categorical_crossentropy", callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    nn_model.fit(X_train_vec,Y_train_vec)

    model_manager = Model_Manager(nn_model)
    model_manager.save_model(path=path)
    precision, recall, fscore, topk = model_manager.calculate_metrics(sentences_vec=X_test_vec, labels_vec=Y_test_vec, labels=Y)

    print("[LSTM]Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore) + " AC@3: " + str(topk) + "\n")


if __name__ == "__main__":
    main()
