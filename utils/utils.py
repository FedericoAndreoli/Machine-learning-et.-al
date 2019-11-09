from difflib import SequenceMatcher
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from wordcloud import WordCloud
import re
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
#import treetaggerwrapper


def correct_dataframe(frame, cat_list):
    """
    Corrects misspelled labels in the dataset.
    :param frame: DataFrame
    :param cat_list: list of categories in the dataframe
    :return: Dataframe with corrected classes
    """
    i = 0
    lung = len(frame)
    printProgressBar(i, lung)
    for cat1 in frame:
        printProgressBar(i, lung)
        i += 1
        for cat2 in cat_list:
            if cat1 != cat2:
                score = similar_sentences(cat1, cat2)
                if score > 0.90 and score != 1:
                    frame.replace(to_replace=cat1, value=cat2)
    return frame


def printProgressBar(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill="■"):  # old █
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write("\r")
        sys.stdout.flush()


def similar_sentences(sentence_1, sentence_2):
    """Compute similarity between two strings
    :param sentence_1: String
    :param sentence_2: String
    :returns sim: similarity score (between 0 and 1)
    """

    sim = SequenceMatcher(None, sentence_1, sentence_2).ratio()
    return sim


def lemmatize_text(text):
    """
    Funtion that lemmatizes input text
    :param text: sentence in a String
    :return: String with lemmatized words in it
    """
    import spacy
    nlp = spacy.load('it_core_news_sm', disable=['parser', 'ner'])
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def clean_text(text, language):
    """
    Cleans text from unwanted words or characters
    :param text: String, sentence
    :param language: on order to select the stopwrods of that particular language
    :return: cleaned text, String
    """
    import string

    # Remove puncuation
    text = text.translate(string.punctuation)
    # Convert words to lower case and split them
    text = text.lower().split()
    # Remove stop words
    stops = set(stopwords.words(language))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"/", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"=", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\W*\b\w{1,2}\b", " ", text)
    text = re.sub(" \d+", " ", text)
    text = re.sub(r"\W*\b\w{26,}\b", " ", text)
    text = re.sub(' +', ' ', text)
    return text


def create_train_test(x, x_labels, test_size=0.25):
    """
    Uses train test split from sk-learn
    :param x: data
    :param x_labels: labels
    :param test_size: size of the testing set
    :return: train data, train labels, test data, test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(x, x_labels, test_size=test_size)

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true.values, y_pred)
    plt.ioff()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.gcf().subplots_adjust(bottom=0.30)
    save_path = os.path.join(os.getcwd(), "Confusion Matrix")
    plt.savefig(os.path.join(save_path, title))

    return ax


def plot_history(history):
    """
    Plot history of loss of a trained neural network
    :param history: history of the training
    :return: /
    """

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def labels_for_NN(dataframe):
    """
    Converts integer labels into one-hot-encoded arrays: 3 --> [0 0 1 0]
    :param dataframe: DataFrame with Label integer field
    :return: list of sparse labels
    """
    num_categories = dataframe.nunique()
    list = dataframe.astype(int)

    NN_labels = np.zeros([1, num_categories])
    for el in list:
        NN_label = np.zeros(num_categories)
        NN_label[el-1] = NN_label[el-1] + 1
        if np.count_nonzero(NN_labels) == 0:
            NN_labels = NN_label
        else:
            NN_labels = np.vstack((NN_labels, NN_label))

    return NN_labels


def convert_data_to_index(list_of_sentences, wv):
    """
    Given a word2vec model compute the embeddings relative to the input
    :param list_of_sentences: sentences to change into vectors
    :param wv: voabulary of the word2vec model (created with Gensim)
    :return: indexed data corresponding to the sentences
    """
    index_data = []
    for sentence in list_of_sentences:
        index_sentence = []
        for word in sentence:
            if word in wv.vocab:
                index_sentence.append(wv.vocab[word].index)
        index_data.append(index_sentence)

    return index_data


def visualize_tfidf_keywords(sentences, vectorizer, title=None, word_cloud=False):

    stop_words = stopwords.words("italian")
    tfidf_bow = vectorizer.transform(sentences)
    df_TMP = pd.DataFrame(tfidf_bow.toarray(), columns=vectorizer.get_feature_names())
    somma = df_TMP.sum().sort_values(ascending=False)
    if word_cloud:
        best = somma[:50].to_dict()
        wordcloud = WordCloud(background_color='white', stopwords=stop_words, max_words=50, max_font_size=50,
                              random_state=42).generate(list(best.keys()))
    else:
        if title is not None:
            plt.title(title)
        somma[:10].plot.bar()  # Select 10 most tfidf-important words
        plt.show()


def report_and_confmat(train_labels, test_labels, prediction, title="generic", save_data=False):

    ytest = np.array(test_labels)

    plot_confusion_matrix(test_labels, prediction, list(set(train_labels)), title=title + " confusion matrix")
    print(title + " results:\n")
    report = classification_report(ytest, prediction, output_dict=False)
    print(report)
    conf_mat = confusion_matrix(ytest, prediction)
    print(conf_mat)
    save_dir = os.path.join(os.getcwd(), "Results")

    if save_data:
        save_npz = os.path.join(save_dir, title + "_arrays")
        np.savez(save_npz, ground_truth=test_labels, prediction=prediction)

    with open(os.path.join(save_dir, title + " results.txt"), "w") as file:
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(conf_mat))



