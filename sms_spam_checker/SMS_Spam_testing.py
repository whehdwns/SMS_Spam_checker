import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from collections import Counter
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, redirect, url_for, request

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

def SMS_testing(message):
    sms_spam_original_dataset = pd.read_csv('spam.csv',encoding='latin-1')
    sms_spam_dataset = sms_spam_original_dataset.copy()

    clean_sms_spam_dataset = sms_spam_dataset.copy()
    clean_sms_spam_dataset.dropna(how="any", inplace=True, axis=1)
    clean_sms_spam_dataset.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

    clean_sms_spam_dataset['label'] = clean_sms_spam_dataset.label.map({'ham':0, 'spam':1})

    clean_sms_spam_dataset['clean_message'] = clean_sms_spam_dataset['message'].apply(clean_text)

    X = clean_sms_spam_dataset.clean_message
    y = clean_sms_spam_dataset.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    countvect = CountVectorizer()
    countvect.fit(X_train)

    X_train_ctvt = countvect.transform(X_train)
    X_train_ctvt = countvect.fit_transform(X_train)
    X_test_ctvt = countvect.transform(X_test)

    model_NB = MultinomialNB()
    model_NB.fit(X_train_ctvt,y_train)

    pred_NB = model_NB.predict(X_test_ctvt)
    NB_score =model_NB.score(X_train_ctvt,y_train)
    NB_pred_score =round(NB_score*100,2)
    #print("MultinomialNB score is " + str(NB_pred_score) +"%")
    NB_pred_accuracy = metrics.accuracy_score(y_test, pred_NB)
    NB_pred_accuracy_score =round(NB_pred_accuracy*100,2)
    #print("MultinomialNB Accuracy of class prediction is " + str(NB_pred_accuracy_score) +"%")
    sms = []
    sms.append(message)

    # sms=["Amazon: Congratulations James, you came 3rd in today's Amazon pods raffle! Click the link to : b2gxv.info/5spfY15YPt",
    #      "May I ask also who this is? I hope I'm texting back the right person I called while ago. Thank you.",
    #      "Final Notice: Dongjun We tried to give your FREE $1000 bonus voucher. Please fill in you details here ki1q.pw/06srYg9ka Enjoy your weekend."
    #     ]

    message=[]

    for i in range(len(sms)):
        review=re.sub('[^a-zA-Z]',' ',sms[i])
        review=review.lower()
        message.append(review)

    predict_sms_using_NB =model_NB.predict(countvect.transform(message))

    if predict_sms_using_NB[0] == 0:
        return "SAFE"
    if predict_sms_using_NB[0] == 1:
        return "Potential SPAM"

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def my_form_post():
    output =""
    if request.method == "POST":
        message = request.form['textarea']
        processed_message = SMS_testing(message)
        output = f"Your message is  { processed_message}"

    return render_template('index.html', output=output)

if __name__ == '__main__':
   app.run(debug = True)
