# Objective 1 – Project 3: MotionAI Sensor System (MQTT + Python + AI Classification)  
**UAT AI Objective:** Create, analyze, and integrate artificial intelligence applications and IoT systems.

---

## 📘 Project Summary

This project builds a cloud-connected motion detection system that uses Python, MQTT, and HiveMQ Cloud to simulate and analyze motion activity. A Python publisher sends motion events (such as detected movement in a camera view) to a cloud MQTT broker, and a Python AI subscriber listens for those events, classifies the activity level (QUIET, NORMAL, or HIGH ACTIVITY), and logs everything to a CSV file.

This project demonstrates how an AI-style decision system can be integrated with an IoT-style messaging pipeline. It shows real-time data flowing from a “sensor” publisher, through the cloud, into an AI analysis layer and then into persistent storage.

---

## 🧱 System Architecture

```text
 ┌─────────────────────────────┐       MQTT Publish        ┌─────────────────────┐
 │ Motion Source (Test Script  │ ───────────────────────▶  │  HiveMQ Cloud Broker │
 │   or Future Camera Input)   │                          │   (TLS, Port 8883)   │
 └─────────────────────────────┘                          └─────────────────────┘
                                                                  │
                                                                  │ MQTT Subscribe
                                                                  ▼
                                                        ┌────────────────────────┐
                                                        │ Motion AI Subscriber    │
                                                        │ - Classifies activity   │
                                                        │ - Logs to CSV           │
                                                        └────────────────────────┘

Key pieces:

    Publisher (test_motion_publish.py)
    Sends JSON motion events to an MQTT topic.

    Cloud Broker (HiveMQ Cloud)
    Handles secure, TLS-encrypted MQTT communication.

    AI Subscriber (motion_ai_subscriber.py)
    Receives messages, classifies motion level, prints results, and logs them.

🧪 MQTT Details

    Broker (Host): your HiveMQ Cloud cluster URL

    Port (Python): 8883 (TLS)

    Port (Web Client): 8884 (WebSocket TLS)

    Topic Used:

    CaribouLouEnterprises/motionSensor1

🧠 AI Classification Logic

The subscriber maintains a rolling window of recent events (for example, the last 20 motion samples). Each event is treated as 1 (motion) or 0 (no motion). It then classifies the environment based on how many motion events occurred in that window.

Example logic:

WINDOW_SIZE = 20
QUIET_THRESHOLD = 3
HIGH_ACTIVITY_THRESHOLD = 10

# recent_events is a deque of 0s and 1s
count = sum(recent_events)

if count <= QUIET_THRESHOLD:
    status = "QUIET"
elif count >= HIGH_ACTIVITY_THRESHOLD:
    status = "HIGH ACTIVITY"
else:
    status = "NORMAL"

This creates a simple AI-style decision layer that transforms raw motion inputs into meaningful activity levels.
📂 Important Files in This Project

Objective1_Project3_MotionAI/
│
├── README.md                      # This file – Boards/overview doc
├── .gitignore                     # Ignores CSV and temporary files
│
├── src/
│   ├── motion_ai_subscriber.py    # AI subscriber (classification + CSV logging)
│   └── test_motion_publish.py     # Test publisher for motion events
│
└── images/
    ├── SS_Test_motion_worked.png      # Screenshot: successful MQTT publish
    ├── SS_MotionDetectionWorked.png   # Screenshot: subscriber classification output
    └── SS_CSVReport.png               # Screenshot: logged motion_events.csv

▶ How to Run the Project
1. Install Dependencies

From your Python environment:

pip install paho-mqtt

(For future camera-based motion detection, you would also install opencv-python, but the current pipeline uses a test publisher.)
2. Configure MQTT Settings

In both src/motion_ai_subscriber.py and src/test_motion_publish.py, confirm these values (using your actual HiveMQ values):

MQTT_BROKER = "your-hivemq-cluster.s1.eu.hivemq.cloud"
MQTT_PORT   = 8883
MQTT_TOPIC  = "CaribouLouEnterprises/motionSensor1"

MQTT_USERNAME = "your-username"
MQTT_PASSWORD = "your-password"

3. Start the AI Subscriber

From the Objective1_Project3_MotionAI/src folder:

python motion_ai_subscriber.py

Expected console output when it connects:

Connecting to HiveMQ...
Listening for motion events... Press CTRL+C to stop.
Connected to HiveMQ.
Subscribed to CaribouLouEnterprises/motionSensor1

When a motion event is received:

Motion Event @ 2025-12-10T18:41:55.385779+00:00
Area: 12345
Activity Level: QUIET
----------------------------------------

4. Publish a Test Motion Event

Open another terminal in the same src folder:

python test_motion_publish.py

This sends a JSON event to the MQTT broker:

{
  "timestamp": "2025-12-10T18:41:55.385779+00:00",
  "motion": true,
  "motion_area": 12345
}

You will see:

    The message appear in the HiveMQ Cloud Web Client if subscribed to the topic.

    The AI subscriber print the motion classification.

    A new row added to the CSV log file.

📊 CSV Logging

The subscriber writes each event to a CSV file (such as motion_events.csv) with fields like:

timestamp, motion, motion_area
2025-12-10T18:41:55.385779+00:00, True, 12345

This log can be used later for further AI/ML experiments, trend analysis, or visualization.
🧾 Screenshots for Boards

The images folder includes screenshots to support Boards documentation, such as:

    MQTT publish success

    Subscriber classification output

    CSV log showing multiple motion events

These screenshots help demonstrate that the system is actually running end-to-end.
🎯 How This Project Meets UAT AI Objective 1

    Objective 1: Create, analyze and integrate artificial intelligence applications and IoT systems.

This project directly supports Objective 1 by:

    Creating an AI-style classification system that interprets motion data and labels the environment as QUIET, NORMAL, or HIGH ACTIVITY.

    Analyzing real-time sensor-like input sent over MQTT and turning it into useful, human-readable information.

    Integrating multiple technologies into one pipeline:

        An IoT-style publisher (simulated motion source in Python),

        A secure cloud-based MQTT broker (HiveMQ Cloud),

        An AI subscriber that performs real-time classification and logging.

Altogether, this shows that I can design, build, and document an integrated AI + IoT system that behaves like a real-world smart sensor environment and is ready to present for UAT Boards.
