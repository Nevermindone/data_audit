import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import linear_model
from time import time

#решенные и нерешенные проблемы
df_neresh = pd.read_csv('C:/Users/best/Desktop/otzivi/otzivi_neresh.csv', delimiter=";")
df_resh = pd.read_csv('C:/Users/best/Desktop/otzivi/otzivi_resh.csv', delimiter=";")
df=pd.concat([df_neresh,df_resh])
#11 labels out of 20
df.loc[df['vid'] == 'Отсутствие воды ', 'vid'] = 'Вода'
df.loc[df['vid'] == 'Протечка кровли (крыши)', 'vid'] = 'Проблемы, связанные с крышей'
df.loc[df['vid'] == 'Игнорирование обращений в управляющие компании', 'vid'] = 'Игнорирование обращений в УК'
df.loc[df['vid'] == 'Снег и наледь', 'vid'] = 'Плохое состояние двора'
df.loc[df['vid'] == 'Не вывозят мусор', 'vid'] = 'Плохое состояние двора'
df.loc[df['vid'] == 'Подъезды в плохом состоянии ', 'vid'] = 'Плохое состояние подъездов'
df.loc[df['vid'] == 'Ошибки в квитанциях', 'vid'] = 'Проблемы с квитанциями'
df.loc[df['vid'] == 'Отсутствие отопления', 'vid'] = 'Отопление'
df.loc[df['vid'] == 'Антисанитария в жилых домах', 'vid'] = 'Антисанитария'
df.loc[df['vid'] == 'Отсутствие капитального ремонта', 'vid'] = 'Плохое состояние подъездов'
df.loc[df['vid'] == 'Неисправный лифт', 'vid'] = 'Лифт'
df.loc[df['vid'] == 'Насекомые и грызуны', 'vid'] = 'Антисанитария'
df.loc[df['vid'] == 'Плачевное состояние дворов', 'vid'] = 'Плохое состояние двора'
df.loc[df['vid'] == 'Грязь в подъезде и лифте', 'vid'] = 'Плохое состояние подъездов'
df.loc[df['vid'] == 'Подъезды в плохом состоянии', 'vid'] = 'Плохое состояние подъездов'
df.loc[df['vid'] == 'Проблемы с поставщиками электроэнергии', 'vid'] = 'Проблемы с электроэнергией'
df.loc[df['vid'] == 'Грязь во дворах', 'vid'] = 'Плохое состояние двора'
df.loc[df['vid'] == 'Низкая температура горячей воды', 'vid'] = 'Вода'
df.loc[df['vid'] == 'Дополнительные платежи в управляющие компании', 'vid'] = 'Проблемы с квитанциями'
df.loc[df['vid'] == 'Нарушения при предоставлении информации от управляющей организации', 'vid'] = 'Нарушения при предоставлении информации от УК'
df.loc[df['vid'] == 'Угроза падения льда с крыш зданий', 'vid'] = 'Проблемы, связанные с крышей' 

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
from nltk.corpus import stopwords
stopwords = set(stopwords.words("russian"))

count_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words=stopwords, max_features=1500) 
train_data_features = count_vectorizer.fit_transform(train_data['text'])
def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data['text'])
    predictions = classifier.predict(data_features)
    target = data['vid']
    evaluate_prediction(predictions, target)
def evaluate_prediction(predictions, target, title="Confusion matrix"):
    print('accuracy %s' % accuracy_score(target, predictions))
    cm = confusion_matrix(target, predictions)
    print('confusion matrix\n %s' % cm)
    print('(row=expected, col=predicted)')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title + ' Normalized')
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(11)
    target_names = ['Вода','Плохое состояние двора','Плохое состояние подъездов','Игнорирование обращений в УК',
                    'Отопление','Лифт','Антисанитария','Проблемы, связанные с крышей','Проблемы с электроэнергией',
                    'Проблемы с квитанциями','Нарушения при предост. инф. от УК']
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


logreg = linear_model.LogisticRegression(n_jobs=3, C=1e5)
logreg = logreg.fit(train_data_features, train_data['vid'])
predict(count_vectorizer, logreg, test_data)
