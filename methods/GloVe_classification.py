from utils import utils
import os
import numpy as np
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, LSTM, Dropout, Bidirectional
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def GloVe_classification(train, valid, labels_train, labels_valid, num_classes, vocabulary, embedding_dim=300, num_epochs=10):

    VOCAB_SIZE = len(vocabulary)
    MAX_SEQUENCE_LENGTH = 800
    EMBEDDING_DIM = embedding_dim

    train_lab = utils.labels_for_NN(labels_train)
    # train_lab = labels_train

    # train_tokens = []
    # list_tot = []
    # for sentence in train:
    #     train_tokens.append(sentence.split())
    #
    # test_tokens = []
    # for sentence in valid:
    #     test_tokens.append(sentence.split())
    #

    # Reading pretrained word embeddings
    embeddings_index = dict()
    path = r'D:\Personal Storage\Federico\Humatics\Text_analysis\FULL DATA\glove_word_embeddings'
    f = open(os.path.join(path, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'), "rb")
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    # In teoria questa versione dovrebbe cercare le parole contenute nel dizionario calcolato dai dati di training
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # idx = vocabulary.get(word)
        embeddings_index[word] = coefs
        # if idx != 0 and idx is not None:
        #     embeddings_index[word] = coefs
    f.close()

    # Create an embedding matrix --> prende i primi n vettori del file?
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train)
    sequences = tokenizer.texts_to_sequences(train)  # --> trasforma parole nei rispettivi indici del dizionario
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, index in tokenizer.word_index.items():
        if index > VOCAB_SIZE - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    # create model
    model_glove = Sequential()
    model_glove.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False))
    model_glove.add(Conv1D(512, 5, activation='sigmoid'))
    model_glove.add(MaxPooling1D(pool_size=4))
    model_glove.add(Bidirectional(LSTM(600, return_sequences=False)))
    model_glove.add(Dense(100, activation='sigmoid'))
    model_glove.add(Dense(num_classes, activation='softmax'))
    model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit train data
    history = model_glove.fit(data, np.array(train_lab), validation_split=0.2, epochs=num_epochs, batch_size=128)
    utils.plot_history(history)

    sequences = tokenizer.texts_to_sequences(valid)
    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    list_prediction_proba = model_glove.predict(data_test)

    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]

    utils.report_and_confmat(labels_train, labels_valid, predizione, "NN_glove" + str(EMBEDDING_DIM))
