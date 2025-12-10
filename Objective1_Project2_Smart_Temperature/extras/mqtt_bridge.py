import serial
import json
import paho.mqtt.client as mqtt

# ----- Serial Settings -----
ser = serial.Serial('COM3', 9600, timeout=2)  # CHANGE COM PORT IF NEEDED

# ----- MQTT Broker -----
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("MQTT Bridge Running... Press CTRL + C to exit.")

try:
    while True:
        line = ser.readline().decode("utf-8").strip()
        if len(line) == 0:
            continue

        print("Serial:", line)

        # Only publish if it's valid JSON
        if line.startswith("{") and line.endswith("}"):
            client.publish(MQTT_TOPIC, line)
            print("Published:", line)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    client.loop_stop()
    client.disconnect()
    ser.close()
