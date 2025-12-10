import paho.mqtt.client as mqtt
import json
from collections import deque
import csv
from datetime import datetime
import os

# ----- CSV Setup -----
csv_file = open("temperature_data.csv", "a", newline="")
csv_writer = csv.writer(csv_file)

# Write header only if file is empty
if os.stat("temperature_data.csv").st_size == 0:
    csv_writer.writerow(["timestamp", "tempC"])

# ----- MQTT Settings -----
MQTT_BROKER = "c8183e9cc06b4e46887fe10381dc0859.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"

MQTT_USERNAME = "mcaruana"
MQTT_PASSWORD = "YOUR_SECURE_PASSWORD"   

# ----- AI Rolling Average -----
recent = deque(maxlen=10)

def classify(current, avg):
    if current > 27 or avg > 26.5:
        return "ALERT (Too Hot)"
    elif current >= 24:
        return "WARM"
    else:
        return "NORMAL"

def on_connect(c, u, f, rc, p=None):
    print("Connected to HiveMQ.")
    c.subscribe(MQTT_TOPIC)
    print(f"Subscribed to {MQTT_TOPIC}")

def on_message(c, u, msg):
    payload = msg.payload.decode()
    print("\nIncoming:", payload)

    try:
        data = json.loads(payload)
        temp = float(data["tempC"])
    except:
        print("Malformed JSON")
        return

    recent.append(temp)
    avg = sum(recent) / len(recent)

    status = classify(temp, avg)

    print(f"Current: {temp:.2f} °C | Avg: {avg:.2f} °C | Status: {status}")

    # ----- Log to CSV -----
    timestamp = datetime.now().isoformat()
    csv_writer.writerow([timestamp, temp])
    csv_file.flush()

# ----- MQTT Client -----
client = mqtt.Client()
client.tls_set()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)

# ----- Start Listening -----
client.loop_forever()

