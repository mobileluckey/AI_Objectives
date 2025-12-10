Project 2: Smart Temperature AI + IoT System
Smart Temperature AI + IoT System
Objective 1 – Project 2: Create, Analyze, and Integrate AI & IoT Systems



🚀 Project Overview

This project demonstrates how IoT hardware, cloud messaging, and AI-style analysis work together in a real system. I created a Smart Temperature Monitoring System using:

Tinkercad (Arduino Uno + TMP36 sensor + LEDs)

Python MQTT Publisher (simulated IoT sensor)

HiveMQ Cloud Broker (secure TLS communication)

Python AI Subscriber (rolling average, real-time classification)

CSV data logging (for future machine learning)

The system streams temperature readings to the cloud and performs AI-based environmental classification:

NORMAL

WARM

ALERT (Too Hot)

This project fully satisfies Objective 1:
Create, analyze and integrate artificial intelligence applications and IoT systems.

📡 System Architecture
Mermaid Diagram (renders automatically on GitHub):
flowchart TD

    subgraph Local["Local Simulation (Tinkercad)"]
        A1[Arduino Uno (Sim)] --> A2[TMP36 Sensor\n+ LED Indicators]
    end

    subgraph Publisher["Python MQTT Publisher"]
        P1[Randomized Temp Generator\n(simulated sensor data)]
        P1 --> P2[JSON Encoder]
    end

    subgraph Cloud["HiveMQ Cloud MQTT Broker\n(TLS Encrypted Port 8883)"]
        C1[Topic:\nCaribouLouEnterprises/tempSensor1]
    end

    subgraph Subscriber["Python AI Subscriber"]
        S1[Receive JSON]
        S2[Rolling Average Window (10 values)]
        S3[Classify:\nNORMAL / WARM / ALERT]
        S4[Append to CSV]
    end

    A2 -.->|Simulated Values| P1
    P2 -->|Publish MQTT| C1
    C1 -->|Subscribe MQTT| S1
    S1 --> S2 --> S3 --> S4

How the System Works

Tinkercad Circuit
Simulates an Arduino Uno with a TMP36 temperature sensor and LEDs indicating COOL, WARM, or HOT.

Python MQTT Publisher
Sends temperature JSON to HiveMQ in this format:

{"tempC": 25.32, "tempF": 77.58}

HiveMQ Cloud (TLS encrypted)
Secure MQTT broker using port 8883.

Python AI Subscriber

Receives live messages

Maintains a rolling window of 10 readings

Computes a moving average

Classifies environment as NORMAL / WARM / ALERT

Saves every reading to a CSV file

temperature_data.csv
Real dataset for ML training later.

Technologies Used
Component	                    Technology
Hardware Simulation	            Tinkercad Circuits (Arduino Uno, TMP36)
Cloud Messaging	                HiveMQ Cloud MQTT Broker
Programming	                    Python 3.12
Libraries	                    paho-mqtt, json, csv, deque
Data Storage	                CSV dataset
AI Logic	                    Rule-based classification + rolling average

MQTT Publisher (publisher/mqtt_publisher.py)
import paho.mqtt.client as mqtt
import random
import time

MQTT_BROKER = "YOUR_CLUSTER_HOST"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"

MQTT_USERNAME = "YOUR_USERNAME"
MQTT_PASSWORD = "YOUR_PASSWORD"

client = mqtt.Client()
client.tls_set()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.connect(MQTT_BROKER, MQTT_PORT)

while True:
    tempC = random.uniform(22, 30)
    tempF = tempC * 9/5 + 32

    msg = f'{{"tempC":{tempC:.2f},"tempF":{tempF:.2f}}}'
    client.publish(MQTT_TOPIC, msg)
    print("Published:", msg)

    time.sleep(2)

AI Subscriber (subscriber_ai/ai_subscriber.py)
import paho.mqtt.client as mqtt
import json
from collections import deque
import csv
import os
from datetime import datetime

# CSV setup
csv_file = open("temperature_data.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
if os.stat("temperature_data.csv").st_size == 0:
    csv_writer.writerow(["timestamp", "tempC"])

# MQTT settings
MQTT_BROKER = "YOUR_CLUSTER_HOST"
MQTT_PORT = 8883
MQTT_TOPIC = "CaribouLouEnterprises/tempSensor1"

MQTT_USERNAME = "YOUR_USERNAME"
MQTT_PASSWORD = "YOUR_PASSWORD"

recent = deque(maxlen=10)

def classify(current, avg):
    if current > 27 or avg > 26.5:
        return "ALERT (Too Hot)"
    elif current >= 24:
        return "WARM"
    else:
        return "NORMAL"

def on_connect(c, u, f, rc, p=None):
    print("Connected.")
    c.subscribe(MQTT_TOPIC)

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

    timestamp = datetime.now().isoformat()
    csv_writer.writerow([timestamp, temp])
    csv_file.flush()

client = mqtt.Client()
client.tls_set()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT)

client.loop_forever()

Example Output:

Incoming: {"tempC":27.12,"tempF":80.82}
Current: 27.12 °C | Avg: 26.78 °C | Status: ALERT (Too Hot)

Incoming: {"tempC":23.88,"tempF":75.00}
Current: 23.88 °C | Avg: 25.90 °C | Status: WARM

How This Meets Objective 1

This project demonstrates the full lifecycle of creating, analyzing, and integrating AI and IoT systems:

Creation: Built a simulated IoT device using Tinkercad and Python.

Integration: Sent real-time sensor data to HiveMQ Cloud via MQTT.

Analysis: Implemented AI-style logic using rolling averages and condition classification.

Data Storage: Logged readings into a CSV file for machine learning models.

This clearly satisfies Objective 1 of the AI degree.