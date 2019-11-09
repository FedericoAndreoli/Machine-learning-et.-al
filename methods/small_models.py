from utils import utils
from utils.utils import *
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, LSTM, Dropout, Bidirectional, Flatten, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from gensim .models import word2vec


batch_size = 32

def create_embedding(train, valid):
    train_clean = train
    train_tokens = []

    list_tot = []
    for sentence in train_clean:
        train_tokens.append(sentence.split())

    test_tokens = []
    for sentence in valid:
        test_tokens.append(sentence.split())

    # 300 = lunghezza del word embedding
    # 600 = lunghezza dle dizionario

    EMBEDDING_DIM = 300

    # USING GENSIM
    # Qui train deve essere una lista di parole, non di stringhe
    model = word2vec.Word2Vec(train_tokens, iter=10, min_count=10, size=EMBEDDING_DIM, workers=4)
    vocab = model.wv.vocab
    VOCAB_SIZE = len(model.wv.vocab)
    MAX_SEQUENCE_LENGTH = 750

    # Mi calcolo i word embedding dei dati di training
    train_sequences = utils.convert_data_to_index(train_tokens, model.wv)
    test_sequences = utils.convert_data_to_index(test_tokens, model.wv)

    # Faccio padding così porto tutti i vettori ad avere la stessa dimensione
    train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocab, train_data, test_data

def conv_classification(train, valid, labels_train, labels_valid, num_classes):

    train_lab = labels_for_NN(labels_train)
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 750
    embedding_matrix, vocab, train_we, test_we = create_embedding(train, valid)
    VOCAB_SIZE = len(vocab)

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix]))
    model.add(Dropout(0.2))
    model.add(Conv1D(512, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # NB binary classification -->binary_crossentropy, Multi-class classification --> categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_we, np.array(train_lab), validation_split=0.2, epochs=5, batch_size=batch_size)
    utils.plot_history(history)

    # SE LA MATRICE TFIDF NON VA BENE O I BAG OF WORDS NON VANNO BENE ALLORA USO QUESTO
    # tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    # sequences = tokenizer.texts_to_sequences(valid)
    # data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # list_prediction_proba = model.predict(data_test)

    list_prediction_proba = model.predict(test_we)

    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]

    utils.report_and_confmat(labels_train, labels_valid, predizione, "TINY_conv_1_layer" + str(EMBEDDING_DIM))


def bi_lstm_classification(train, valid, labels_train, labels_valid, num_classes):

    train_lab = labels_for_NN(labels_train)

    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 750
    embedding_matrix, vocab, train_we, test_we = create_embedding(train, valid)
    VOCAB_SIZE = len(vocab)

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix]))
    model.add(Bidirectional(LSTM(512, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ## Fit train data
    history = model.fit(train_we, np.array(train_lab), validation_split=0.2, epochs=3, batch_size=batch_size)
    utils.plot_history(history)

    # SE LA MATRICE TFIDF NON VA BENE O I BAG OF WORDS NON VANNO BENE ALLORA USO QUESTO
    # tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    # sequences = tokenizer.texts_to_sequences(valid)
    # data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # list_prediction_proba = model.predict(data_test)

    list_prediction_proba = model.predict(test_we)

    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]

    utils.report_and_confmat(labels_train, labels_valid, predizione, "TINY_bilstm" + str(EMBEDDING_DIM))


def lstm_classification(train, valid, labels_train, labels_valid, num_classes):

    train_lab = labels_for_NN(labels_train)
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 750
    embedding_matrix, vocab, train_we, test_we = create_embedding(train, valid)
    VOCAB_SIZE = len(vocab)

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix]))
    model.add(LSTM(512))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ## Fit train data
    history = model.fit(train_we, np.array(train_lab), validation_split=0.2, epochs=3, batch_size=batch_size)
    utils.plot_history(history)

    # SE LA MATRICE TFIDF NON VA BENE O I BAG OF WORDS NON VANNO BENE ALLORA USO QUESTO
    # tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    # sequences = tokenizer.texts_to_sequences(valid)
    # data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # list_prediction_proba = model.predict(data_test)

    list_prediction_proba = model.predict(test_we)

    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]

    utils.report_and_confmat(labels_train, labels_valid, predizione, "TINY_lstm_" + str(EMBEDDING_DIM))