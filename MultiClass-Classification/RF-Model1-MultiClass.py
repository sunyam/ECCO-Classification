# NOTE: To run for different words, change 'numberOfFeatures':
numberOfFeatures = 100

'''
Different hyper-parameters in Random Forest.
'''

from sklearn.ensemble import RandomForestClassifier

def my_RF(X, labels):

    tuned_parameters = [{'n_estimators': [10, 100, 1000], 'max_features': ['auto', 'sqrt', 'log2'],
                         'criterion': ['gini','entropy']}]

    metrics = ['accuracy']
    m = ['accuracy']

    models = []

    for score in metrics:
        model = {}
        rf = RandomForestClassifier()
        clf = GridSearchCV(rf, tuned_parameters, cv=10, scoring=score, verbose=2)
        clf.fit(X, labels)
        print("\nBest parameters: ", str(clf.best_estimator_))
        print("Best score achieved: ", str(clf.best_score_))
        best_rf = clf.best_estimator_
        # Now that I have the best parameters for each metric, running SVM for those specific parameters to obtain
        # all values.
        for s in m:
            model[s] = np.array(cross_val_score(best_rf, X, labels, cv=10, scoring=s))

        print(model)
        models.append(model)

    return models

'''
Load CSVs and getting it ready for CountVectorizer.
'''
import pandas as pd

# categories = ['Agriculture.csv', 'Biography.csv', 'Botany.csv', 'Church.csv', 'Commerce.csv', 'Dictionaries.csv',
#               'Drama.csv', 'Fiction.csv', 'History.csv', 'History_Natural.csv', 'Law.csv', 'Mathematics.csv',
#               'Medicine.csv', 'Physics.csv', 'Poetry.csv', 'Politics.csv', 'Rhetoric.csv', 'Sermons.csv',
#               'Travels.csv']

df_agri = pd.read_csv('./5_features/Model1_length1/agri_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

df_botany = pd.read_csv('./5_features/Model1_length1/botany_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_church = pd.read_csv('./5_features/Model1_lenttgth1/church_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_commerce = pd.read_csv('./5_features/Model1_length1/commerce_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_drama = pd.read_csv('./5_features/Model1_length1/drama_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_fiction = pd.read_csv('./5_features/Model1_length1/fiction_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_history = pd.read_csv('./5_features/Model1_length1/history_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_historyNatural = pd.read_csv('./5_features/Model1_length1/historyNatural_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_law = pd.read_csv('./5_features/Model1_length1/law_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_math = pd.read_csv('./5_features/Model1_length1/math_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_med = pd.read_csv('./5_features/Model1_length1/med_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_phy = pd.read_csv('./5_features/Model1_length1/phy_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_poetry = pd.read_csv('./5_features/Model1_length1/poetry_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_politics = pd.read_csv('./5_features/Model1_length1/politics_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_rhetoric = pd.read_csv('./5_features/Model1_length1/rhetoric_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_sermons = pd.read_csv('./5_features/Model1_length1/sermons_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_travels = pd.read_csv('./5_features/Model1_length1/travels_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
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

vectorizer500 = CountVectorizer()
X500 = vectorizer500.fit_transform(everything)
print("Vectorizer: ", X500.shape) # Prints (57231, something)



results_100 = my_RF(X500, labels)

print(results_100)
