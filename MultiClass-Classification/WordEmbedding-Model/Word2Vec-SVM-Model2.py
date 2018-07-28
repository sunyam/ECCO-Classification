
'''
Different hyper-parameters in SVM.
'''

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV


def my_SVM(X, labels):

    tuned_parameters = [{'C': [1, 10, 1000], 'gamma': [0.001], 'kernel': ['rbf']}]

    metrics = ['accuracy']
    m = ['accuracy']

    models = []

    for score in metrics:
        model = {}
        svc = SVC()

        clf = GridSearchCV(svc, tuned_parameters, cv=10, scoring=score, verbose=5)
        clf.fit(X, labels)
        print("\nBest parameters: ", str(clf.best_estimator_))
        print("Best score achieved: ", str(clf.best_score_))
        best_svc = clf.best_estimator_
        # Now that I have the best parameters for each metric, running SVM for those specific parameters to obtain
        # all values.
        for s in m:
            model[s] = np.array(cross_val_score(best_svc, X, labels, cv=10, scoring=s))

        print(model)
        models.append(model)
    return models




'''
Load CSVs and getting it ready for CountVectorizer.
'''
import pandas as pd

numberOfFeatures = 1000

# categories = ['Agriculture.csv', 'Biography.csv', 'Botany.csv', 'Church.csv', 'Commerce.csv', 'Dictionaries.csv',
#               'Drama.csv', 'Fiction.csv', 'History.csv', 'History_Natural.csv', 'Law.csv', 'Mathematics.csv',
#               'Medicine.csv', 'Physics.csv', 'Poetry.csv', 'Politics.csv', 'Rhetoric.csv', 'Sermons.csv',
#               'Travels.csv']

df_agri = pd.read_csv('./DocClass5.0/5_features/Model2_length3/agri_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

df_botany = pd.read_csv('./DocClass5.0/5_features/Model2_length3/botany_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_church = pd.read_csv('./DocClass5.0/5_features/Model2_length3/church_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_commerce = pd.read_csv('./DocClass5.0/5_features/Model2_length3/commerce_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_drama = pd.read_csv('./DocClass5.0/5_features/Model2_length3/drama_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_fiction = pd.read_csv('./DocClass5.0/5_features/Model2_length3/fiction_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_history = pd.read_csv('./DocClass5.0/5_features/Model2_length3/history_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_historyNatural = pd.read_csv('./DocClass5.0/5_features/Model2_length3/historyNatural_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_law = pd.read_csv('./DocClass5.0/5_features/Model2_length3/law_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_math = pd.read_csv('./DocClass5.0/5_features/Model2_length3/math_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_med = pd.read_csv('./DocClass5.0/5_features/Model2_length3/med_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_phy = pd.read_csv('./DocClass5.0/5_features/Model2_length3/phy_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_poetry = pd.read_csv('./DocClass5.0/5_features/Model2_length3/poetry_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_politics = pd.read_csv('./DocClass5.0/5_features/Model2_length3/politics_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_rhetoric = pd.read_csv('./DocClass5.0/5_features/Model2_length3/rhetoric_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_sermons = pd.read_csv('./DocClass5.0/5_features/Model2_length3/sermons_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_travels = pd.read_csv('./DocClass5.0/5_features/Model2_length3/travels_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

# Removing NaN
df_agri.fillna('', inplace=True)

df_botany.fillna('', inplace=True)
df_church.fillna('', inplace=True)
df_commerce.fillna('', inplace=True)
df_drama.fillna('', inplace=True)
df_fiction.fillna('', inplace=True)
df_history.fillna('', inplace=True)
df_historyNatural.fillna('', inplace=True)
df_law.fillna('', inplace=True)
df_math.fillna('', inplace=True)
df_med.fillna('', inplace=True)
df_phy.fillna('', inplace=True)
df_poetry.fillna('', inplace=True)
df_politics.fillna('', inplace=True)
df_rhetoric.fillna('', inplace=True)
df_sermons.fillna('', inplace=True)
df_travels.fillna('', inplace=True)

# Changing it to CountVec fashion:
agri = [' '.join(str(r) for r in row) for row in df_agri.values]

botany = [' '.join(str(r) for r in row) for row in df_botany.values]
church = [' '.join(str(r) for r in row) for row in df_church.values]
commerce = [' '.join(str(r) for r in row) for row in df_commerce.values]

drama = [' '.join(str(r) for r in row) for row in df_drama.values]
fiction = [' '.join(str(r) for r in row) for row in df_fiction.values]
history = [' '.join(str(r) for r in row) for row in df_history.values]
historyNatural = [' '.join(str(r) for r in row) for row in df_historyNatural.values]
law = [' '.join(str(r) for r in row) for row in df_law.values]
math = [' '.join(str(r) for r in row) for row in df_math.values]
med = [' '.join(str(r) for r in row) for row in df_med.values]

phy = [' '.join(str(r) for r in row) for row in df_phy.values]
poetry = [' '.join(str(r) for r in row) for row in df_poetry.values]
politics = [' '.join(str(r) for r in row) for row in df_politics.values]
rhetoric = [' '.join(str(r) for r in row) for row in df_rhetoric.values]
sermons = [' '.join(str(r) for r in row) for row in df_sermons.values]
travels = [' '.join(str(r) for r in row) for row in df_travels.values]

# Passing it to CountVectorizer:
import numpy as np

everything = agri + botany + church + commerce + drama + fiction + history + historyNatural + law + math + med + \
             phy + poetry + politics + rhetoric + sermons + travels
print("Everything: ", len(everything))

# Defining labels (must be in the same order)
labels = len(agri)*['Agriculture'] + len(botany)*['Botany'] + len(church)*['Church'] + len(commerce)*['Commerce'] + len(drama)*['Drama'] + len(fiction)*['Fiction'] + len(history)*['History'] + len(historyNatural)*['History Natural'] + len(law)*['Law'] + len(math)*['Mathematics'] + len(med)*['Medicine'] + len(phy)*['Physics'] + len(poetry)*['Poetry'] + len(politics)*['Politics'] + len(rhetoric)*['Rhetoric'] + len(sermons)*['Sermons'] + len(travels)*['Travels']

# Storing sizes (in same order)
sizes_in_same_order = [len(agri), len(botany), len(church), len(commerce), len(drama), len(fiction), len(history), len(historyNatural), len(law), len(math), len(med), len(phy), len(poetry), len(politics), len(rhetoric), len(sermons), len(travels)]

count = 0
for i in sizes_in_same_order:
    count += i

print("Count: ", count)


import gensim

path = './ECCO_model.word2vec'
model = gensim.models.Word2Vec.load(path)

# Average all word2vec vectors:
# document is a list of 1000 words (Model 2)
def Doc2Vec(document):
    vector = [0.0]*300
    numberOfWords = 0

    for word in document:
        if word in model.wv.vocab:
            numberOfWords += 1
            vec = model[word]
            vector = np.add(vector, vec)

    if numberOfWords != 0:
        avg_vector = np.nan_to_num(vector/numberOfWords)
        return avg_vector

    else:
        print("Oops no words")
        return vector



X = []

for document in df_agri.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_botany.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_church.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_commerce.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_drama.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_fiction.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_history.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_historyNatural.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_law.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_math.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_med.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_phy.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_poetry.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_politics.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_rhetoric.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_sermons.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)


for document in df_travels.values:
    doc_vector = Doc2Vec(document)
    X.append(doc_vector)

print(len(X))

X = np.array(X)

print(X.shape)


results_100 = my_SVM(X, labels)

print(results_100)
