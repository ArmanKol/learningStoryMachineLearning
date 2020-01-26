import requests
import json

URL =  'http://localhost:8080/rest/rules/'

data = {}
actions = {}
triggers = {}

configurationTrigger = {}
configurationAction = {}

triggers['id'] = 2
triggers['label'] = "it is a fixed time of day"
triggers['description'] = "Triggers at a specified time"
configurationTrigger['time'] = "08:00"
triggers['configuration'] = configurationTrigger
triggers['type'] = "timer.TimeOfDayTrigger"

actions['id'] = 3
actions['label'] = "send a command"
actions['description'] = "Sends a command to a specified item."
configurationAction['itemName'] = "bathroomWashingMachine_1_Relay"
configurationAction['command'] = "ON"
actions['configuration'] = configurationAction
actions['type'] = "core.ItemCommandAction"

data['triggers'] = [triggers]
data['actions'] = [actions]
data['name'] = "TESTT"

json_data = json.dumps(data)
requests.post(URL, data=json_data, headers={"content-type": "application/json"})