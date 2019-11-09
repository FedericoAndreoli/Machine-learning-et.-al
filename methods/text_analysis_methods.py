from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.feature_extraction.text import TfidfVectorizer
from utilities.db_handler import *
from word_embeddings import create_word_embedding
import pickle as pkl
from utilities.utils import *
from multiprocessing import Process, Manager
import json
import os
import numpy as np
import pandas as pd


def random_forest_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data iterable/list
    :param train_labels: training labels
    :param test_labels: testing labels
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with Random Forest Classifier...")
    rand = RandomForestClassifier(n_estimators=70, max_depth=None)
    rand.fit(train, train_labels)

    prediction = rand.predict(test)
    report_and_confmat(test_labels, prediction, "Random Forest")
    score = rand.score(test, test_labels)

    res["RandomForestClassifier"] = {"parameters": rand.get_params(), "accuracy": score, "name": "RandomForestClassifier"}
    print("RandomForset ended...")
    return score, rand


def SVC_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with SVC...")

    svc = SVC(kernel='poly', gamma='scale')
    svc.fit(train, train_labels)

    prediction = svc.predict(test)
    report_and_confmat(test_labels, prediction, "SVC")
    score = svc.score(test, test_labels)

    res["SVC"] = {"parameters": svc.get_params(), "accuracy": score, "name": "SVC"}
    print("SVC ended...")
    return score, svc


def LinearSVC_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with LinearSVC...")

    linear_svc = LinearSVC()
    linear_svc.fit(train, train_labels)

    prediction = linear_svc.predict(test)
    report_and_confmat(test_labels, prediction, "LinearSVC")
    score = linear_svc.score(test, test_labels)

    res["LinearSVC"] = {"parameters": linear_svc.get_params(), "accuracy": score, "name": "LinearSVC"}
    print("LinearSVC ended...")
    return score, linear_svc


def MultinomialNB_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    multiNB = MultinomialNB()
    multiNB.fit(train, train_labels)

    prediction = multiNB.predict(test)
    report_and_confmat(test_labels, prediction, "MultinomialNB")
    score = multiNB.score(test, test_labels)

    res["MultinomialNB"] = {"parameters": multiNB.get_params(), "accuracy": score, "name": "MultinomialNB"}
    print("Multinomial ended...")
    return score, multiNB


def ComplementNB_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with Complement Nive Bayes...")

    complNB = ComplementNB()
    complNB.fit(train, train_labels)

    prediction = complNB.predict(test)
    report_and_confmat(test_labels, prediction, "ComplementNB")
    score = complNB.score(test, test_labels)

    res["ComplementNB"] = {"parameters": complNB.get_params(), "accuracy": score, "name": "ComplementNB"}
    print("Complement ended...")
    return score, complNB


def BernoulliNB_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with Bernoulli Nive Bayes...")

    bernNB = BernoulliNB(alpha=0.7)
    bernNB.fit(train, train_labels)

    prediction = bernNB.predict(test)
    report_and_confmat(test_labels, prediction, "BernoulliNB")
    score = bernNB.score(test, test_labels)
    res["BernoulliNB"] = {"parameters": bernNB.get_params(), "accuracy": score, "name": "BernoulliNB"}
    print("Bernoulli ended...")

    return score, bernNB


def GradientBoosting_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with Gradient Boosting...")

    gradb = GradientBoostingClassifier(n_estimators=100)
    gradb.fit(train, train_labels)

    prediction = gradb.predict(test)
    report_and_confmat(test_labels, prediction, "GradientBoosting")
    score = gradb.score(test, test_labels)
    res["GradientBoostingClassifier"] = {"parameters": gradb.get_params(), "accuracy": score, "name": "GradientBoostingClassifier"}
    print("GradientBoosting ended...")

    return score, gradb


def AdaBoost_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with AdaBoost...")

    # Uso l'svc perché è quello che funziona meglio per ora
    Linsvc = LinearSVC()

    adab = AdaBoostClassifier(base_estimator=Linsvc, algorithm='SAMME', n_estimators=50)
    adab.fit(train, train_labels)

    prediction = adab.predict(test)
    report_and_confmat(test_labels, prediction, "AdaBoost")
    score = adab.score(test, test_labels)
    print("Adaboost ended...")
    res["AdaBoostClassifier"] = {"parameters": adab.get_params(), "accuracy": score, "name": "AdaBoostClassifier"}

    return score, adab


def VotingClassifier_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with Voting classifier...")

    cl1 = LogisticRegression(max_iter=250, multi_class='auto')
    cl6 = MultinomialNB()
    cl3 = AdaBoostClassifier(base_estimator=cl1, algorithm='SAMME', n_estimators=150)
    cl4 = GradientBoostingClassifier()
    cl5 = ComplementNB()
    cl8 = RandomForestClassifier(n_estimators=70, max_depth=None)
    cl9 = ExtraTreesClassifier()

    vote = VotingClassifier(estimators=[('LogisticReg', cl1), ('AdaBoost', cl3), ('GradBoost', cl4),
                            ('ComplementNB', cl5), ('MultinomialNB', cl6), ('RandomForest', cl8),
                            ('ExtraTree', cl9)], voting='soft')
    vote.fit(train, train_labels)

    prediction = vote.predict(test)
    report_and_confmat(test_labels, prediction, "VotingClass")
    score = vote.score(test, test_labels)
    print("Voting ended...")
    res["VotingClassifier"] = {"parameters": vote.get_params(), "accuracy": score, "name": "VotingClassifier"}

    return score, vote


def LogisticRegression_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """

    print("Classifying with LogisticRegression...")

    # TODO CONTROLLARE I SOLVER DIVERSI
    reg = LogisticRegression(max_iter=250, multi_class='multinomial', solver='newton-cg')
    reg.fit(train, train_labels)

    prediction = reg.predict(test)
    report_and_confmat(test_labels, prediction, "LogisticReg")
    score = reg.score(test, test_labels)

    res["LogisticRegression"] = {"parameters": reg.get_params(), "accuracy": score, "name": "LogisticRegression"}
    print("Logistic Regression ended...")
    return score, reg


def ExtrExtraTrees_classification(train, test, train_labels, test_labels, res={}):
    """

    :param train: training data, iterable/list
    :param test: testing data, iterable/list
    :param train_labels: training labels, iterable/list
    :param test_labels: testing labels, iterable/list
    :return: / --> Saves data in folder "Results"
    """
    print("Classifying with ExtraTrees...")

    extra = ExtraTreesClassifier()
    extra.fit(train, train_labels)
    prediction = extra.predict(test)

    report_and_confmat(test_labels, prediction, "ExtraTrees")
    score = extra.score(test, test_labels)

    res["ExtraTrees"] = {"parameters": extra.get_params(), "accuracy": score, "name": "ExtraTreesClassifier"}
    print("ExtraTrees ended...")
    return score, extra


def classifiers_pipeline(train_bow, test_bow, label_train, label_test, save_path):
    """
    Calls all the classifiers functions in order to choose and save the best one.
    :param train_bow: training BagOfWords, iterable/list
    :param test_bow: testing BagOfWords, iterable/list
    :param label_train: training labels, iterable/list
    :param label_test: testing labels, iterable/list
    :param save_path: (fixed to Models directory)
    :return: /
    """
    best = 0
    best_model = None
    name = 'None'

    rand_score, rand_model = random_forest_classification(train_bow, test_bow, label_train, label_test)
    if rand_score > best:
        name = 'RandomForestClassifier'
        best = rand_score
        best_model = rand_model

    svc_score, svc_model = SVC_classification(train_bow, test_bow, label_train, label_test)
    if svc_score > best:
        name = 'SVC'
        best = svc_score
        best_model = svc_model

    lin_svc_score, lin_svc_model = LinearSVC_classification(train_bow, test_bow, label_train, label_test)
    if lin_svc_score > best:
        name = 'LinearSVC'
        best = lin_svc_score
        best_model = lin_svc_model

    multiNB_score, multiNB_model = MultinomialNB_classification(train_bow, test_bow, label_train, label_test)
    if multiNB_score > best:
        name = 'MultinomialNB'
        best = multiNB_score
        best_model = multiNB_model

    complNB_score, complNB_model = ComplementNB_classification(train_bow, test_bow, label_train, label_test)
    if complNB_score > best:
        name = 'ComplementNB'
        best = complNB_score
        best_model = complNB_model

    bernNB_score, bernNB_model = BernoulliNB_classification(train_bow, test_bow, label_train, label_test)
    if bernNB_score > best:
        name = 'BernoulliNB'
        best = bernNB_score
        best_model = bernNB_model

    gradboost_score, gradboost_model = GradientBoosting_classification(train_bow, test_bow, label_train, label_test)
    if gradboost_score > best:
        name = 'GradientBoostingClassifier'
        best = gradboost_score
        best_model = gradboost_model

    logReg_score, logReg_model = LogisticRegression_classification(train_bow, test_bow, label_train, label_test)
    if logReg_score > best:
        name = 'LogisticRegression'
        best = logReg_score
        best_model = logReg_model

    adaBoost_score, adaBoost_model = AdaBoost_classification(train_bow, test_bow, label_train, label_test)
    if adaBoost_score > best:
        name = 'AdaBoostClassifier'
        best = adaBoost_score
        best_model = adaBoost_model

    # voting_score, voting_model = VotingClassifier_classification(train_bow, test_bow, label_train, label_test)
    # if voting_score > best:
    #     name = 'VotingClassifier'
    #     best = voting_score
    #     best_model = voting_model

    extraTree_score, extraTree_model = ExtrExtraTrees_classification(train_bow, test_bow, label_train, label_test)
    if extraTree_score > best:
        name = 'ExtraTreesClassifier'
        best = extraTree_score
        best_model = extraTree_model

    data = {"parameters": best_model.get_params(), "accuracy": best, "name": name}
    model_name = name + "_parameters.json"
    save_dir = os.path.join(save_path, model_name)
    try:
        with open(save_dir, "wb") as pklfile:
            pkl.dump(data, pklfile)
    except RuntimeError as e:
        print("Error occurred when saving the model:", e)


def parallel_pipeline(train_bow, test_bow, label_train, label_test, save_path):
    import threading
    manager = Manager()
    return_dict = manager.dict()

    # Use a shared variable in order to get the results
    proc = []
    fncs1 = [random_forest_classification, SVC_classification, LinearSVC_classification, MultinomialNB_classification,
            LogisticRegression_classification, BernoulliNB_classification, GradientBoosting_classification,
            AdaBoost_classification, ComplementNB_classification, VotingClassifier_classification,
            ExtrExtraTrees_classification]

    # instantiating process with arguments
    for fn in fncs1:
        p = threading.Thread(target=fn, args=(train_bow, test_bow, label_train, label_test, return_dict))
        proc.append(p)
        p.start()
    for p in proc:
        p.join()

    # Save the best model's parameters into a pkl file
    best = 0
    for dict in return_dict.values():
        if dict["accuracy"] > best:
            best = dict["accuracy"]
            name = dict["name"]
            model_parameters = dict["parameters"]

    data = {"parameters": model_parameters, "accuracy": best, "name": name}
    filename = os.path.join(save_path, name + "_data.pkl")
    with open(filename, "wb") as pklfile:
        pkl.dump(data, pklfile)


def train_classifiers(data_path, save_path, multithreading, lang='eng'):

    # Read data from provided location
    dataframe = pd.read_csv(data_path, header=0, sep=";")
    dataframe = dataframe.dropna()

    print("Cleaning text...", end="")
    dataframe['text'] = dataframe['text'].map(lambda x: clean_text(x, lang))
    print("Done")

    #Train test split
    train, test, label_train, label_test = create_train_test(dataframe["text"], dataframe["label"])

    # TFIDF Bag Of Words extraction. Theese lines extract the features from text based on word frequencies among
    # texts
    print("Creating TFIDF data")
    unwanted = stopwords.words(lang)
    tfidf_vect = TfidfVectorizer(analyzer='word', ngram_range=range(1, 3), max_features=3000, stop_words=unwanted,
                                 max_df=0.5, min_df=3)
    tfidf_vect.fit(train)

    # saving the vectorizer model
    tmp = os.path.join(save_path, "Models")
    save = os.path.join(tmp, "vectorizer.pkl")
    with open(save, 'wb') as pklfile:
        pkl.dump(tfidf_vect, pklfile)

    train_bow = tfidf_vect.transform(train)
    test_bow = tfidf_vect.transform(test)

    print("Beginning supervised training...")
    if multithreading:
        parallel_pipeline(train_bow, test_bow, label_train, label_test, save_path)
    else:
        classifiers_pipeline(train_bow, test_bow, label_train, label_test, save_path)


def predict_label(data_path):
    """
    Predicts the label of a given ticket. Given the ticket_id it gets the text from the database and then it saves the
    result in the database
    :param ticket_id: integer corresponding to the ticket to classify
    :return: /
    """
    try:
        for fname in os.listdir(os.getcwd()):
            if "data.pkl" in fname:
                with open(fname, "rb") as file:
                    model_dict = pkl.load(file)
                    model_param = model_dict["parameters"]
                    model_name = model_dict["name"]
                    # Se non può ritornare la confidenza delle classi uso l'accuratezza come indice
                    confidence = model_dict["accuracy"]
            # else:
            #     print("Error when searching for model file")
            #     quit(-1)
            if "vectorizer" in fname:
                with open(fname, "rb"):
                    vect_param = pkl.load(fname)
            # else:
            #     raise FileNotFoundError("Error when searching for vectorizer file")

        # Load the parameters into each object
        vectorizer = TfidfVectorizer()
        vectorizer.set_params(**vect_param)
        model = eval(model_name+"()")
        model.set_params(**model_param)

        # Create text and clean it from useless words
        try:
            data["text"] = data["ticket_title"].map(str) + ' ' + data["description"].map(str)
            data["text"] = data['text'].map(lambda x: clean_text(x))
        except ValueError as e:
            print("Error when processing the text.")
            print(e)

        data_bow = vectorizer.transform(data["text"])
        prediction = model.predict(data_bow)

        # This part computes the prediction confidence but there's no field in the database.
        if "LinearSVC" in model_dict["name"]:
            # LinearSVC doesn't support the predict_proba method. Use model accuracy intesad.
            tmp = np.ones(len(data_bow))
            scores = np.dot(tmp, confidence)
        else:
            scores = model.predict_proba(data_bow)

        result = dict()
        result["prediction"] = prediction
        result["score"] = scores
        to_write = json.dumps(result)

        # save_predictions_in_db(ticket_id, prediction)

        save_predictions_in_db(ticket_id, to_write)

    except RuntimeError as e:
        print("Error when making prediction")
        print(e)
