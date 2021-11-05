from pprint import pprint
import numpy as np
# from sklearn.linear_model import LinearRegression
import mlflow
import pandas as pd
from nltk.corpus import stopwords
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
experiment_name = "modeltextlabel"
mlflow.set_experiment(experiment_name)


df=pd.read_csv('imagetxt1.csv')

stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ['eee','good','number','malaysia','kuala lampur','china','ee','asean','trade good agreement',
             'place date signature','code','cee',]
stop_words = stop_words.union(new_words)

corpus = []
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

for i in range(0, len(df)):
    #     try :

    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;'/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ##Convert to list from string
    text = text.split()

    ##Stemming
    ps = PorterStemmer()
    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
                                                        stop_words]
    text = " ".join(text)
    corpus.append(text)


train,test = train_test_split(df,test_size=0.1,shuffle=True,random_state=42)
X_train = train.text
X_test = test.text

model  = Pipeline([
    ('tfidf',TfidfVectorizer(ngram_range=(1, 3),max_df=130,min_df=20)),
    ('clf',SGDClassifier(loss='hinge')),
])

import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, \
    plot_confusion_matrix

import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, \
    plot_confusion_matrix

with mlflow.start_run(nested=True) as run:
    # Build the model
    tic = time.time()
    #     model = SGDClassifier(random_state=123,loss='log')
    model.fit(X_train, train['label'])
    duration_training = time.time() - tic

    # Make the prediction
    tic1 = time.time()
    prediction = model.predict(X_test)
    duration_prediction = time.time() - tic1

    # Evaluate the model prediction
    metrics = {
        # "rmse": np.sqrt(mean_squared_error(test['label'], prediction)),
        # "mae": mean_absolute_error(test['label'], prediction),
        # "r2": r2_score(test['label'], prediction),
        "duration_training": duration_training,
        "duration_prediction": duration_prediction

    }

    # Log in mlflow (parameter)
    #     mlflow.log_params(parameters)

    # Log in mlflow (metrics)
    mlflow.log_metrics(metrics)
    # confuise matrix
    acc = accuracy_score(test['label'], prediction)
    confusion_matrices = confusion_matrix(test['label'], prediction)

    disp = plot_confusion_matrix(model, X_test, test['label'],
                                 cmap=plt.cm.Blues)
    filename = 'text.png'
    disp.figure_.savefig(filename)

    #     roc = metrics.roc_auc_score(test['label'], prediction)
    # confusion matrix values
    tp = confusion_matrices[0][0]
    tn = confusion_matrices[1][1]
    fp = confusion_matrices[0][1]
    fn = confusion_matrices[1][0]
    # get classification metrics
    class_report = classification_report(test['label'], prediction, output_dict=True)
    recall_invoice = class_report['invoice']['recall']
    f1_score_invoice = class_report['invoice']['f1-score']
    recall_pack = class_report['packing list']['recall']
    f1_score_pack = class_report['packing list']['f1-score']

    mlflow.log_metric("accuracy_score", acc)
    mlflow.log_metric("true_positive", tp)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)
    mlflow.log_metric("recall_invoice", recall_invoice)
    mlflow.log_metric("f1_score_invoice", f1_score_invoice)
    mlflow.log_metric("recall_packing_list", recall_pack)
    mlflow.log_metric("f1_score_packing_list", f1_score_pack)
    mlflow.log_artifact(filename, 'confusin_matrix')

    #   log in mlflow (model)
    mlflow.sklearn.log_model(model, f"model")





