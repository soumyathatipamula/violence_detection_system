# **Violence Detection System**

This project leverages deep learning techniques for detecting violent activities in video footage. Using a combination of MobileNetV2 and LSTMs, the system processes sequences of frames to classify videos into "Normal" or "Violence" categories. It also integrates a Telegram bot to send alerts when violence is detected.

## **Features**

- **Deep Learning Model**: Utilizes a TimeDistributed MobileNetV2 backbone with LSTM layers for sequence analysis.
- **Real-Time Processing**: Capable of detecting violence in live or recorded video feeds.
- **Telegram Alerts**: Automatically sends alerts, including frames from the video, to a specified Telegram group.
- **Scalable Framework**: Handles video preprocessing, training, and inference efficiently.

## **Technologies Used**

- **Programming Language**: Python
- **Frameworks and Libraries**:
    - TensorFlow/Keras
    - OpenCV
    - Scikit-learn
    - Telepot (for Telegram bot integration)
- **Models**: MobileNetV2, LSTM
- **Visualization**: Matplotlib

---

## **Dataset Structure**

The project uses the SCVD dataset, structured as follows:

```
plaintext
Copy code
SCVD/
├── Train/
│   ├── Normal/
│   ├── Violence/
│   ├── Weaponized/
├── Test/
    ├── Normal/
    ├── Violence/
    ├── Weaponized/

```

Each subdirectory contains video files of respective categories.

---

## **Installation**

1. Clone the repository:
    
    ```bash
    git clone https://github.com/soumyathatipamula/violence_detection_system.git
    cd violence_detection_system
    
    ```
    
2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
3. Download the SCVD dataset and organize it as described above. [Dataset](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd)
4. Set up a Telegram bot using BotFather and update the bot token and group ID in the script.

---

## **Usage**
1. Run the `cctv-classification.ipynb` file
    
2. The model will be saved to `./Models/violence_detection.h5`.

### **Real-Time Inference with Alerts**

1. Run the flask app:
    
    ```bash
    python Flask_app/app.py
    
    ```
    
2. Ensure the Telegram bot is active to send alerts.
