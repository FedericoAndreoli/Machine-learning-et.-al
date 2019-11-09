from utils import utils
from gensim .models import word2vec
import numpy as np
from keras.layers import Embedding, Conv1D, Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam


def word2vec_classification(train, valid, labels_train, labels_valid, num_classes, num_epochs=10):

    train_lab = utils.labels_for_NN(labels_train)
    # train_lab = labels_train
    # valid_lab = labels_for_NN(labels_valid)

    #    train_clean = utils.clean_df_text(train)

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

    #vocab_size = 730613 --> memory error
    # model = word2vec.Word2Vec.load(r'D:\Personal Storage\Federico\Humatics\Text_analysis\FULL DATA\Dati preprocessati\glove_wiki_window10_size300_iteration50\glove_WIKI')
    # vocab_size = 733392--> memory error
    # model = word2vec.Word2Vec.load(r'D:\Personal Storage\Federico\Humatics\Text_analysis\FULL DATA\Dati preprocessati\word2vec_skipgram_wiki_window10_size300_neg-samples10\wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m')
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

    # COMBINATO
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix]))
    model.add(Conv1D(512, 5, activation='sigmoid'))
    # model.add(GlobalMaxPooling1D())
    model.add(Bidirectional(LSTM(600, return_sequences=False)))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                  metrics=['acc'])
    model.fit(train_data, train_lab, validation_split=0.2, epochs=num_epochs, batch_size=80)

    list_prediction_proba = model.predict(test_data)

    predizione = [np.where(probabilities == probabilities.max())[0].min() for probabilities in list_prediction_proba]

    utils.report_and_confmat(labels_train, labels_valid, predizione, "word2vec_")

    # CODICE ORIGINALE DELLA RETE
    # Per fare learning dei pesi durante il training sostituire il prcedente con:
    # embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix])
    #
    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
    # embedded_sequences = embedding_layer(sequence_input)
    # x = Conv1D(166, 5, activation='relu')(embedded_sequences)
    # x = GlobalMaxPooling1D()(x)
    # x = Flatten()(x)
    # x = Dense(50, activation='relu')(x)
    # preds = Dense(num_classes, activation='softmax')(x)
    #
    # model = Model(sequence_input, preds)
