import paho.mqtt.client as mqtt
import random
import time

# ----- HiveMQ Cloud Settings -----
MQTT_BROKER = "c8183e9cc06b4e46887fe10381dc0859.s1.eu.hivemq.cloud"
MQTT_PORT = 8883               # TLS port
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"   # rename if you like

# >>>>>> PUT YOUR HIVEMQ CLOUD USERNAME/PASSWORD HERE <<<<<<
MQTT_USERNAME = "mcaruana"
MQTT_PASSWORD = "YOUR_SECURE_PASSWORD"

# Create client
client = mqtt.Client()

# Enable TLS (HiveMQ Cloud requires encrypted connection)
client.tls_set()  # uses system CA certs; good enough for HiveMQ Cloud

# Set username/password auth
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Connect to HiveMQ Cloud
client.connect(MQTT_BROKER, MQTT_PORT)

print("Publishing simulated temperature data to HiveMQ Cloud...")
print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
print(f"Topic : {MQTT_TOPIC}")
print("Press Ctrl + C to stop.\n")

try:
    while True:
        # Simulated TMP36-style temperatures
        tempC = random.uniform(22.0, 30.0)
        tempF = tempC * 9.0 / 5.0 + 32.0

        # Build JSON payload
        msg = f'{{"tempC":{tempC:.2f},"tempF":{tempF:.2f}}}'

        # Publish
        result = client.publish(MQTT_TOPIC, msg)
        status = result[0]
        if status == 0:
            print("Sent:", msg)
        else:
            print("Failed to send message")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nSimulation stopped.")

finally:
    client.disconnect()
