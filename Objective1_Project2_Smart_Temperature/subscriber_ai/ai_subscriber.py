import paho.mqtt.client as mqtt
import json
from collections import deque

# ----- HiveMQ Cloud Settings -----
MQTT_BROKER = "c8183e9cc06b4e46887fe10381dc0859.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"

MQTT_USERNAME = "mcaruana"
MQTT_PASSWORD = "YOUR_SECURE_PASSWORD"

# Keep last N readings for a rolling average (simple “learning”/analysis)
window_size = 10
recent_temps = deque(maxlen=window_size)


def classify_temp(current, avg):
    """
    Simple AI-style classification based on current temp
    and rolling average.
    """
    if current is None:
        return "UNKNOWN"

    # You can tweak these thresholds however you like
    if current > 27 or avg > 26.5:
        return "ALERT (Too Hot)"
    elif current >= 24:
        return "WARM"
    else:
        return "NORMAL"


def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected to HiveMQ Cloud with result code", reason_code)
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")


def on_message(client, userdata, msg):
    global recent_temps

    payload = msg.payload.decode("utf-8")
    print("\nRaw message:", payload)

    try:
        data = json.loads(payload)
        tempC = float(data.get("tempC"))
    except Exception as e:
        print("Error parsing JSON:", e)