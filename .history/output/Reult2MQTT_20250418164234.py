import pandas as pd
import paho.mqtt.client as paho
from paho import mqtt
import time
import json
import random 
# File containing forecast data
long_term_csv_file = "./long_term_prediction_results.csv"
short_term_csv_file = "./short_term_alert.json"

# Read the CSV file using pandas
long_term_data = pd.read_csv(long_term_csv_file)




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


# Publish each row's forecast data to the respective topics
for index, row in long_term_data.iterrows():
    if index > 30:
        break 
    # Create payloads (you can include more details like timestamp if needed)
    payload_RF = f"{row['RandomForest_Prediction']}"
    payload_LR = f"{row['LogisticRegression_Prediction']}"

    # Publish to the topics
    client.publish(topic_RF, payload_RF)
    client.publish(topic_LR, payload_LR)

    print(f"Published row {index}: RF={payload_RF}, LR={payload_LR}")

    # Optional: wait a bit between rows
    time.sleep(0.5)

try:
    # Read the JSON file 
    live_data_alerts = None
    with open(short_term_csv_file, "r") as f:
        live_data_alerts = json.load(f) 
    
    for record in live_data_alerts:
        message = json.dumps(record)
        result = client.publish("COMP4436/Project/LIVE", payload=message, qos=1)
        print(f"Publishing to COMP4436/Project/LIVE: {message}")
        time.sleep(0.5)  
except Exception as e:
    print("Error reading or sending live-data alert:", e)



# Disconnect the client after publishing all messages
client.loop_stop()
client.disconnect()