import json
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from random import randrange

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def convert(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)

def opslaanRule():
    ruleGenereren(runKnn(1, 0, 0), "A", "ON")
    ruleGenereren(runKnn(0, 0, 0), "B", "OFF")
    print(runKnn(1, 0, 0))

def ruleGenereren(tijd, ruleAB, status):
    URL = 'http://localhost:8080/rest/rules/'
    randomID = randrange(50000)

    data = {}
    actions = {}
    triggers = {}

    configurationTrigger = {}
    configurationAction = {}

    triggers['id'] = randomID
    triggers['label'] = "it is a fixed time of day"
    triggers['description'] = "Triggers at a specified time"
    configurationTrigger['time'] = tijd
    triggers['configuration'] = configurationTrigger
    triggers['type'] = "timer.TimeOfDayTrigger"

    actions['id'] = randomID+1
    actions['label'] = "send a command"
    actions['description'] = "Sends a command to a specified item."
    configurationAction['itemName'] = "bathroomWashingMachine_1_Relay"
    configurationAction['command'] = status
    actions['configuration'] = configurationAction
    actions['type'] = "core.ItemCommandAction"

    data['triggers'] = [triggers]
    data['actions'] = [actions]
    data['name'] = "MachineLearning"+"_"+ruleAB

    json_data = json.dumps(data)
    requests.post(URL, data=json_data, headers={"content-type": "application/json"})

def runKnn(stand, dag, kamer):
    learningData = pd.read_csv("test.csv")
    learningData['datum'] = learningData['datum'].astype('datetime64')
    learningData['datum'] = learningData['datum'].dt.date

    for j in learningData['stand']:
        if ('ON' in j):
            learningData['stand'] = learningData['stand'].str.replace('ON', '1')
        if ('OFF' in j):
            learningData['stand'] = learningData['stand'].str.replace('OFF', '0')

    learningData = learningData.drop(columns="datum")

    #X = input, Y=output
    X=learningData[['stand', 'dag', 'kamer']]
    y=learningData['seconds']

    # splits in train en test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
    print('Aantal trainwaarden {0:d}'.format(len(X_train)))
    print('Aantal testwaarden {0:d}'.format(len(y_test)))

    # Met hoeveel punten er vergeleken wordt.
    knn = KNeighborsClassifier(n_neighbors=3)
    # Trainen van de data
    knn.fit(X_train, y_train)
    # laat de classifier de testset berekenen
    y_knn = knn.predict(X_test)

    print("Score:")
    print(knn.score(X_train, y_train))

    # voorspel tijd met stand, dag, kamer
    stand_prediction = knn.predict([[stand, dag, kamer]])
    print("Stand_prediction:")

    return convert(stand_prediction[0])

def runDecisionThree(stand, dag, kamer):
    learningData = pd.read_csv("test.csv")
    learningData['datum'] = learningData['datum'].astype('datetime64')
    learningData['datum'] = learningData['datum'].dt.date

    for j in learningData['stand']:
        if ('ON' in j):
            learningData['stand'] = learningData['stand'].str.replace('ON', '1')
        if ('OFF' in j):
            learningData['stand'] = learningData['stand'].str.replace('OFF', '0')

    learningData = learningData.drop(columns="datum")

    # X = input, Y=output
    X = learningData[['stand', 'dag', 'kamer']]
    y = learningData['seconds']

    # splits in train en test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('Aantal trainwaarden {0:d}'.format(len(X_train)))
    print('Aantal testwaarden {0:d}'.format(len(y_test)))


    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree_clf.fit(X, y)

    print("Score:")
    print(tree_clf.score(X_train, y_train))

    # voorspel tijd met stand, dag, kamer
    stand_prediction = tree_clf.predict([[stand, dag, kamer]])

    return convert(stand_prediction[0])

opslaanRule()