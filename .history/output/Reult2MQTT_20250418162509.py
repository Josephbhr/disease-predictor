import pandas as pd
import paho.mqtt.client as mqtt
import paho
import time
import json

# File containing forecast data
long_term_csv_file = "./long_term_prediction_results.csv"
short_term_csv_file = "./short_term_alert.json"

# Read the CSV file using pandas
long_term_data = pd.read_csv(long_term_csv_file)




# MQTT Broker settings
broker = "5b98359e206d4a2abba0bbce72be4559.s1.eu.hivemq.cloud"
port = 8883

# MQTT topics
topic_RF = "COMP4436/Project/RF"
topic_LR = "COMP4436/Project/LR"
topic_live = "COMP4436/Project/LIVE"

# Create an MQTT client instance
client = mqtt.Client()

# Connect to the broker
client.connect(client_id="", userdata=None, protocol=paho.MQTTv5)
client.username_pw_set("YOUR_USERNAME", "YOUR_PASSWORD")  # Replace with your HiveMQ Cloud username/password
client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

# Connect to your HiveMQ Cloud cluster
client.connect("YOUR_CLUSTER_URL", 8883)  # Replace with your cluster URL (e.g. "abc123.s1.eu.hivemq.cloud")

client.loop_start()


# Publish each row's forecast data to the respective topics
for index, row in long_term_data.iterrows():
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
    with open(short_term_csv_file, "r") as f:
        live_data_records = json.load(f) 

except Exception as e:
    print("Error reading or sending live-data:", e)



# Disconnect the client after publishing all messages
client.loop_stop()
client.disconnect()