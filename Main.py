import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
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
    ruleGenereren(runDecisionThree(1, 0, 0), "A", "ON")
    ruleGenereren(runDecisionThree(0, 0, 0), "B", "OFF")

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

def runKnn():
    learningData = pd.read_csv("test.csv")

    learningData['datum'] = learningData['datum'].astype('datetime64')
    learningData['tijd'] = learningData['datum'].dt.time
    learningData['datum'] = learningData['datum'].dt.date
    learningData['tijd'] = learningData['tijd'].astype('str')
    print(learningData.dtypes)

    seconds = []

    for i in learningData['tijd']:
        seconds.append(get_sec(i))

    for j in learningData['stand']:
        if ('ON' in j):
            learningData['stand'] = learningData['stand'].str.replace('ON', '1')
        if ('OFF' in j):
            learningData['stand'] = learningData['stand'].str.replace('OFF', '0')

    for k in learningData['datum']:
        learningData['dag'] = k.weekday()+1


    learningData['seconds'] = seconds
    learningData = learningData.drop(columns="datum")

    dict_switch = dict(zip(learningData['id'].unique(),learningData['stand'].unique()))
    print("dict_switch:")
    print(dict_switch)

    print(learningData.head())
    #X = input, Y=output
    X=learningData[['stand', 'dag', 'kamer']]
    y=learningData['seconds']

    # splits in train en test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
    print('Aantal trainwaarden {0:d}'.format(len(X_train)))
    print('Aantal testwaarden {0:d}'.format(len(y_test)))

    k_range = range(1,26)
    scores = {}
    scores_list = []

    for k in k_range:
        # Het indelen van groepen x aantal groepen
        knn = KNeighborsClassifier(n_neighbors=2)
        # Trainen van de data
        knn.fit(X_train, y_train)
        # laat de classifier de testset berekenen
        y_knn = knn.predict(X_test)
        # controleer de nauwkeurigheid met de test data
        scores[k] = metrics.accuracy_score(y_test, y_knn)
        scores_list.append(metrics.accuracy_score(y_test, y_knn))

    plt.plot(k_range, scores_list)
    plt.xlabel('value of K for KNN')
    plt.ylabel('Tesint Accuracy')
    #plt.show()

    # controleer met een confusion matrix
    cm = confusion_matrix(y_test,y_knn)
    print(cm)

    # voorspel voor een bepaalde seconds, dag, kamer
    stand_prediction = knn.predict([[0, 0, 0]])
    print("Stand_prediction:")

    #print(dict_switch[stand_prediction[0]])
    print(convert(stand_prediction[0]))

def runDecisionThree(status, dag, kamer):
    learningData = pd.read_csv("test.csv")

    learningData['datum'] = learningData['datum'].astype('datetime64')
    learningData['tijd'] = learningData['datum'].dt.time
    learningData['datum'] = learningData['datum'].dt.date
    learningData['tijd'] = learningData['tijd'].astype('str')

    seconds = []

    for i in learningData['tijd']:
        seconds.append(get_sec(i))

    for j in learningData['stand']:
        if ('ON' in j):
            learningData['stand'] = learningData['stand'].str.replace('ON', '1')
        if ('OFF' in j):
            learningData['stand'] = learningData['stand'].str.replace('OFF', '0')

    for k in learningData['datum']:
        learningData['dag'] = k.weekday() + 1

    learningData['seconds'] = seconds
    learningData = learningData.drop(columns="datum")

    # X = input, Y=output
    X = learningData[['stand', 'dag', 'kamer']]
    y = learningData['seconds']

    # splits in train en test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('Aantal trainwaarden {0:d}'.format(len(X_train)))
    print('Aantal testwaarden {0:d}'.format(len(y_test)))

    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)

    print("Score:")
    print(tree_clf.score(X_train, y_train))

    # voorspel tijd met stand, dag, kamer
    stand_prediction = tree_clf.predict([[status, dag, kamer]])

    return convert(stand_prediction[0])

opslaanRule()