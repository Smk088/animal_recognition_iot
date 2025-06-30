# AI-WildNet

**An Automated System for Real-Time Wildlife Monitoring and Early Warning at Road Crossings**



## 📌 Project Overview

AI-WildNet is a cost-effective, AI-powered system designed to detect and deter wildlife in real-time, specifically at 
road crossings and farmlands. It combines image recognition using Convolutional Neural Networks (CNN), motion sensors, and ultrasonic deterrents 
to minimize human-wildlife conflict — without requiring expensive GPS modules.

## 🧠 Key Features

* 🦌 Real-time wildlife detection using cameras and CNN-based image classification.
* 📡 Motion detection via PIR sensors.
* 📍 GPS-free SMS alerts using static location tags.
* 🔊 Non-invasive ultrasonic animal deterrent system.
* 🌐 Flask-based web interface for monitoring, alerts, and logs.
* 💻 Low-cost IoT hardware for deployment in rural environments.

---

## 📁 Project Structure

```bash
AI-WildNet/
│
├── static/                      # Static web assets (CSS, JS)
├── templates/                  # Flask HTML templates
├── model/                      # Trained CNN model (WildNet)
├── dataset/                    # Training dataset (e.g., BRA Dataset)
├── iot/                        # ESP8266, sensor code, and schematics
├── app.py                      # Flask web server
├── animal_predictor.py         # CNN/TCN inference scripts
├── utils.py                    # Helper utilities (e.g., SMS, logging)
├── requirements.txt            # Python dependencies
└── README.md                   # Project description
```

---

## 🔧 Technologies Used

* **Programming Languages:** Python, Arduino
* **Libraries:** OpenCV, TensorFlow, Flask, Scikit-learn
* **Hardware:** ESP8266, Ultrasonic Sensor, IR Sensor, Servo Motor
* **Database:** MySQL
* **Web Framework:** Flask + Bootstrap
* **Dataset:** [Brazilian Road’s Animals Dataset (BRA-Dataset)](#dataset)

---

## 📷 System Architecture

The AI-WildNet system consists of the following key modules:

* **Hardware Unit:** ESP8266-based IoT kit with sensors and camera
* **Software Unit:**

  * CNN-based image classification (WildNet model)
  * Temporal Convolutional Network (TCN) for behavior prediction
  * Flask dashboard for admins and forest rangers
* **Alert System:** Real-time SMS alerts and ultrasonic buzzer

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/AI-WildNet.git
cd AI-WildNet
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask App

```bash
python app.py
```

### 4. Upload Trained Model

Place your trained `wildnet_model.h5` into the `model/` directory.

---

## 🧪 Dataset

The system uses the **Brazilian Road’s Animals Dataset (BRA-Dataset)**, consisting of 1823 images of endangered Brazilian species frequently seen on roadsides.

* Image format: `.jpg`
* Annotation: VOC format
* Classes: Tapir, Puma, Maned Wolf, Jaguarundi, Giant Anteater

---

## 📊 Performance Metrics

* **Accuracy:** \~92%
* **Precision, Recall, F1 Score:** Evaluated using confusion matrix
* **Hardware Efficiency:** Lightweight, low-power design for rural deployment

---

## 🚀 Future Enhancements

* Integration with IR/night-vision cameras
* Mobile app for local users
* Solar-powered kits
* Cloud-based data analytics
* Multi-language support for alerts (e.g., Tamil)

---

## 👨‍💻 Authors


* S. Mohanakrishna
* G. Gokul
* R. Praveenkumar
* M. Vasanthakumar


Developed as part of the final year B.E. Computer Science & Engineering project at **M.P. Nachimuthu M. Jaganathan Engineering College**, affiliated with **Anna University**, Chennai.

---
