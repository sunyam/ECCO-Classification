'''
Load CSVs and getting it ready for CountVectorizer.
'''
import pandas as pd

numberOfFeatures = 1000

# categories = ['Agriculture.csv', 'Biography.csv', 'Botany.csv', 'Church.csv', 'Commerce.csv', 'Dictionaries.csv',
#               'Drama.csv', 'Fiction.csv', 'History.csv', 'History_Natural.csv', 'Law.csv', 'Mathematics.csv',
#               'Medicine.csv', 'Physics.csv', 'Poetry.csv', 'Politics.csv', 'Rhetoric.csv', 'Sermons.csv',
#               'Travels.csv']

df_agri = pd.read_csv('../DocClass5.0/5_features/Model2_length3/agri_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

df_botany = pd.read_csv('../DocClass5.0/5_features/Model2_length3/botany_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_church = pd.read_csv('../DocClass5.0/5_features/Model2_length3/church_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_commerce = pd.read_csv('../DocClass5.0/5_features/Model2_length3/commerce_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_drama = pd.read_csv('../DocClass5.0/5_features/Model2_length3/drama_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_fiction = pd.read_csv('../DocClass5.0/5_features/Model2_length3/fiction_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_history = pd.read_csv('../DocClass5.0/5_features/Model2_length3/history_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_historyNatural = pd.read_csv('../DocClass5.0/5_features/Model2_length3/historyNatural_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_law = pd.read_csv('../DocClass5.0/5_features/Model2_length3/law_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_math = pd.read_csv('../DocClass5.0/5_features/Model2_length3/math_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_med = pd.read_csv('../DocClass5.0/5_features/Model2_length3/med_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_phy = pd.read_csv('../DocClass5.0/5_features/Model2_length3/phy_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_poetry = pd.read_csv('../DocClass5.0/5_features/Model2_length3/poetry_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_politics = pd.read_csv('../DocClass5.0/5_features/Model2_length3/politics_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_rhetoric = pd.read_csv('../DocClass5.0/5_features/Model2_length3/rhetoric_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_sermons = pd.read_csv('../DocClass5.0/5_features/Model2_length3/sermons_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))
df_travels = pd.read_csv('../DocClass5.0/5_features/Model2_length3/travels_' + str(numberOfFeatures) + '.csv', header=None, names=range(0,numberOfFeatures))

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
from sklearn.feature_extraction.text import CountVectorizer
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
print("\nTrain SVM on all 57k data..")

# Training SVM on all 57k data

from sklearn.svm import SVC

svm = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=True)

print(X500.shape)
print(len(labels))
svm.fit(X500, labels)




'''
Now getting the Test Data ready:
'''
# Predicting on 98k files:
import pickle
with open('./features_ECCO2_52k.pickle', 'rb') as f:
    test_features = pickle.load(f)
print(len(test_features))

# Convert test_features dict into two lists (preserving order):
fnames = []
wordLists = []
for (filename, wordList) in test_features.items():
    fnames.append(filename)
    wordLists.append(wordList)
print(len(fnames))
print(len(wordLists))

listOfStrings = [' '.join(str(word) for word in LIST) for LIST in wordLists]
print(len(listOfStrings))

# Pass it to CountVec now:
X_test = vectorizer500.transform(listOfStrings)
print(X_test.shape)

print("Now predicting....")
PREDICTIONS = svm.predict(X_test)
print(len(PREDICTIONS))

# Map the predictions to filenames:

FINAL_OUTPUT = {}

for fname,pred in zip(fnames, PREDICTIONS):
    FINAL_OUTPUT[fname] = pred

print("Now Pickling..")
output_path = '/Users/dh_lab_05/Desktop/ECCO2_Map_FileToLabel.pickle'
with open(output_path, 'wb') as f:
    pickle.dump(FINAL_OUTPUT, f)
