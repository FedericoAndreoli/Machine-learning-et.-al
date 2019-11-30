from utils import utils
from gensim .models import word2vec
import numpy as np
from keras.layers import Embedding, Conv1D, Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam


def word2vec_classification(train, valid, labels_train, labels_valid, save_path, num_classes, num_epochs=10):
    """
    Calls all the classifiers functions in order to choose and save the best one.
    :param train: training data, iterable/list
    :param valid: testing data, iterable/list
    :param label_train: training labels, iterable/list
    :param label_test: testing labels, iterable/list
    :param num_classes: number of classes in training data, integer
    :param num_epochs=10: number of epochs to perform, integer
    :param save_path: (fixed to Models directory)
    :return: /
    """
    train_lab = utils.labels_for_NN(labels_train)

    train_clean = train
    train_tokens = []

    list_tot = []
    for sentence in train_clean:
        train_tokens.append(sentence.split())

    test_tokens = []
    for sentence in valid:
        test_tokens.append(sentence.split())

    # Dimension of the embedding vector representing the words
    EMBEDDING_DIM = 300

    # USING GENSIM, it needs a list of training TOKENS and it builds the vocabulary
    model = word2vec.Word2Vec(train_tokens, iter=10, min_count=10, size=EMBEDDING_DIM, workers=4)
    VOCAB_SIZE = len(model.wv.vocab)
    MAX_SEQUENCE_LENGTH = 750

    # Compute training embedding
    train_sequences = utils.convert_data_to_index(train_tokens, model.wv)
    test_sequences = utils.convert_data_to_index(test_tokens, model.wv)

    # Pad the vectors so they're all the same length
    train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

    # Getting the embedding matrix, a lookup table that translates a known word into a vector
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Build a network: made out of first convolutional part and second recurrent part (LSTM)
    # NB Thenetwork is very small and basic because of strict system requirements
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix]))
    model.add(Conv1D(512, 5, activation='sigmoid'))
    model.add(GlobalMaxPooling1D())
    model.add(Bidirectional(LSTM(600, return_sequences=False)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                  metrics=['acc'])

    # Train the nwtwork
    model.fit(train_data, train_lab, validation_split=0.2, epochs=num_epochs, batch_size=80)

    # Make predictions
    list_prediction_proba = model.predict(test_data)

    # Compute report and confusion matrix
    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]
    utils.report_and_confmat(labels_train, labels_valid, predizione, save_path, "word2vec_")
