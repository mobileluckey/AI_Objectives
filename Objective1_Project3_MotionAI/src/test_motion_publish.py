import paho.mqtt.client as mqtt
import json
from datetime import datetime, UTC

MQTT_BROKER = "a75c2adaf1d448eaae0b0c2313b25024.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/motionSensor1"

MQTT_USERNAME = "mcaruana"
MQTT_PASSWORD = "password"

client = mqtt.Client()
client.tls_set()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

print("Connecting to MQTT broker...")
client.connect(MQTT_BROKER, MQTT_PORT)

event = {
    "timestamp": datetime.now(UTC).isoformat(),
    "motion": True,
    "motion_area": 12345
}

payload = json.dumps(event)

result = client.publish(MQTT_TOPIC, payload)
status = result[0]

if status == 0:
    print("Published test motion event:", payload)
else:
    print("Failed to publish message.")
