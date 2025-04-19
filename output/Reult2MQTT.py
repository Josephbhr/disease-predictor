import pandas as pd
import paho.mqtt.client as paho
from paho import mqtt
import time
import json
import random 
# File containing forecast data

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
long_term_json_file  = os.path.join(BASE_DIR, "long_term_report.json")
short_term_json_file = os.path.join(BASE_DIR, "short_term_alert.json")




# MQTT Broker settings (Use HiveMQ Cloud as Learned in IC Labs)
MQTT_CLUSTER_URL = "5b98359e206d4a2abba0bbce72be4559.s1.eu.hivemq.cloud"
USERNAME = "Joseph"
PASSWORD = "COMP4436!Great"

# MQTT topics
topic_RF = "COMP4436/Project/RF"
topic_LR = "COMP4436/Project/LR"
topic_live = "COMP4436/Project/LIVE"
client_id = f'COMP4436-Project-{random.randint(0, 100)}'
client = paho.Client(client_id = client_id, userdata=None, protocol=paho.MQTTv5)


# Connect to the broker
client.username_pw_set(USERNAME, PASSWORD)  
client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

# Connect to your HiveMQ Cloud cluster
client.connect(MQTT_CLUSTER_URL, port = 8883) 

client.loop_start()



try:
    with open(long_term_json_file, "r") as f:
        payload = json.load(f)
        message = json.dumps(payload)
        client.publish("COMP4436/Project/LONGTERM", payload=message, qos=1)
        print(f"Publishing to COMP4436/Project/LONGTERM: {message}")
except Exception as e:
    print("Error reading or sending long-term analysis:", e)



try:
    # Read the JSON file 
    live_data_alerts = None
    with open(short_term_json_file, "r") as f:
        live_data_alerts = json.load(f) 
    
    for record in live_data_alerts:
        message = json.dumps(record)
        result = client.publish("COMP4436/Project/LIVE", payload=message, qos=1)
        print(f"Publishing to COMP4436/Project/LIVE: {message}")
        time.sleep(1.5)  
except Exception as e:
    print("Error reading or sending live-data alert:", e)



# Disconnect the client after publishing all messages
client.loop_stop()
client.disconnect()