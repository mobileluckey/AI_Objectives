import paho.mqtt.client as mqtt
import json
from collections import deque
from datetime import datetime, UTC
import csv
import os

# ---------- MQTT SETTINGS ----------
MQTT_BROKER = "a75c2adaf1d448eaae0b0c2313b25024.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/motionSensor1"

MQTT_USERNAME = "mcaruana"
MQTT_PASSWORD = "password"

# ---------- ANALYSIS SETTINGS ----------
WINDOW_SIZE = 20
QUIET_THRESHOLD = 3
HIGH_ACTIVITY_THRESHOLD = 10

recent_events = deque(maxlen=WINDOW_SIZE)

# ---------- CSV LOGGING ----------
CSV_FILENAME = "motion_events.csv"

def setup_csv_writer():
    file_exists = os.path.isfile(CSV_FILENAME)
    csv_file = open(CSV_FILENAME, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists or os.stat(CSV_FILENAME).st_size == 0:
        writer.writerow(["timestamp", "motion", "motion_area"])

    return csv_file, writer

csv_file, csv_writer = setup_csv_writer()

def classify_activity():
    count = sum(recent_events)

    if count <= QUIET_THRESHOLD:
        return "QUIET"
    elif count >= HIGH_ACTIVITY_THRESHOLD:
        return "HIGH ACTIVITY"
    else:
        return "NORMAL"

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to HiveMQ.")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print("Connection failed with code:", rc)

def on_message(client, userdata, msg):
    global csv_writer, csv_file

    try:
        data = json.loads(msg.payload.decode())
        motion = data.get("motion", False)
        area = data.get("motion_area", 0)
        timestamp = data.get("timestamp", datetime.now(UTC).isoformat())
    except Exception as e:
        print("Malformed JSON:", e)
        return

    recent_events.append(1 if motion else 0)

    status = classify_activity()

    print(f"\nMotion Event @ {timestamp}")
    print(f"Area: {area}")
    print(f"Activity Level: {status}")
    print("-" * 40)

    csv_writer.writerow([timestamp, motion, area])
    csv_file.flush()

def main():
    client = mqtt.Client()
    client.tls_set()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting to HiveMQ...")
    client.connect(MQTT_BROKER, MQTT_PORT)

    print("Listening for motion events... Press CTRL+C to stop.")
    client.loop_forever()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
        csv_file.close()
