Objective 1 – Project 3: MotionAI Sensor System (MQTT + Python + AI Classification)
UAT AI Objective: Create, analyze, and integrate artificial intelligence applications and IoT systems.
📌 Project Summary

This project uses a motion-detecting AI system built with Python, MQTT, and a cloud broker (HiveMQ). The system simulates motion events from a camera source, publishes them to the cloud, and classifies the environment as QUIET, NORMAL, or HIGH ACTIVITY using a simple rolling-window AI logic model. This demonstrates the ability to integrate IoT devices, cloud messaging, data classification, and real-time AI-powered feedback.

🎯 What This Project Demonstrates for Objective 1

This project proves that I can:

Build an AI-driven IoT system using cloud-based messaging.

Process real-time sensor input using Python and publish it to the cloud.

Classify data using an AI-style rule-based model (threshold-based inference).

Store sensor data in a CSV file for later AI/ML analysis.

Run both a publisher (sensor simulator) and subscriber (AI engine).

This is a fully working example of creating, analyzing, and integrating AI and IoT systems, which is exactly what Objective 1 requires.

🗂️ System Architecture
Camera / Motion Algorithm
        |
        v
Python Motion Publisher  ---->  MQTT Cloud Broker (HiveMQ)  ---->
                                           |
                                           v
                         AI Motion Subscriber (Classification + Logging)

🧠 AI Classification Logic

The subscriber maintains the last 20 motion-events:

0 = no motion

1 = motion detected

Then it classifies:

Condition	Status
≤ 3 motion events	QUIET
4–9 motion events	NORMAL
≥ 10 motion events	HIGH ACTIVITY

This is a lightweight inference model appropriate for IoT devices with limited compute.

📡 MQTT Topic Used
CaribouLouEnterprises/motionSensor1

🧪 Sample Published JSON Event
{
  "timestamp": "2025-12-10T18:41:55.385779+00:00",
  "motion": true,
  "motion_area": 12345
}

🖥️ Python Scripts Included
1. Motion Publisher (test_motion_publish.py)

Simulates the camera sending motion data.

result = client.publish(MQTT_TOPIC, payload)

2. Motion AI Subscriber (motion_ai_subscriber.py)

Receives events, classifies activity, prints results, and logs CSV data.

Activity Level: QUIET / NORMAL / HIGH ACTIVITY

📁 CSV Logging

All events are saved for later model training:

motion_events.csv


Fields:

timestamp

motion (true/false)

motion_area (numeric heatmap region)

How This Project Meets Objective 1

Objective 1: Create, analyze, and integrate artificial intelligence applications and IoT systems.

This project fully satisfies Objective 1 because it shows that I can design an AI-driven IoT system from end to end. 
I created a motion-detection pipeline that uses Python to generate sensor events, publishes those events to a real 
cloud MQTT broker, and then analyzes the incoming data with an AI-style classification layer. The subscriber script 
keeps a rolling history of events, evaluates real-time conditions, classifies the environment as QUIET, NORMAL, or 
HIGH ACTIVITY, and logs all data for future model training.

This demonstrates that I can create an intelligent system, analyze sensor patterns, integrate cloud services, and 
build an IoT architecture that behaves like a real deployment. It proves I can combine AI logic, Python automation,
networking concepts, and IoT communication into one working solution.

