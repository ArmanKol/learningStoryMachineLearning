import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)


learningData = pd.read_csv("test.csv")
df = pd.DataFrame(learningData)
origineel = pd.DataFrame(learningData)
print(df.head())

df['datum'] = df['datum'].astype('datetime64')
df['tijd'] = df['datum'].dt.time
df['datum'] = df['datum'].dt.date
df['tijd'] = df['tijd'].astype('str')
print(df.dtypes)

seconds = []

for i in df['tijd']:
    seconds.append(get_sec(i))

for j in df['stand']:
    if ('ON' in j):
        df['stand'] = df['stand'].str.replace('ON', '1')
    if ('OFF' in j):
        df['stand'] = df['stand'].str.replace('OFF', '0')

df['seconds'] = seconds

dict_switch = dict(zip(learningData['id'].unique(),learningData['stand'].unique()))
dict_switch

#X = input, Y=output
X=learningData[['stand','kamer']]
y=learningData['seconds']

# splits in train en test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
print('Aantal trainwaarden {0:d}'.format(len(X_train)))
print('Aantal testwaarden {0:d}'.format(len(y_test)))

# Het indelen van groepen x aantal groepen
knn = KNeighborsClassifier(n_neighbors = 5)
# Trainen van de data
knn.fit(X_train,y_train)
# laat de classifier de testset berekenen
y_knn = knn.predict(X_test)
# controleer de nauwkeurigheid met de test data
knn.score(X_test,y_test)

# controleer met een confusion matrix
cm = confusion_matrix(y_test,y_knn)
print(cm)

# voorspel voor een bepaalde vrucht
stand_prediction = knn.predict([[0,0]])
stand_prediction

print(convert(stand_prediction[0]))

data = {}
data['key'] = 'value'
json_data = json.dumps(data)