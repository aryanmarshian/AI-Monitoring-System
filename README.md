## 🛡️ Smart Surveillance & Drone Detection System

This is a real-time AI-powered surveillance system capable of detecting **humans**, **animals**, and **drones** using YOLOv8 and Vision Transformers. It captures frames from a live video feed, enhances image quality, detects objects, sends email alerts, saves data to MongoDB, and optionally identifies camera location via IP-based geolocation.

 

## 🚀 Features

- ✅ Human & Animal detection using YOLOv8 general model
- ✅ Dedicated Drone detection model with fallback to general model
- ✅ Image enhancement (wavelet denoising, contrast correction)
- ✅ Real-time object detection via webcam or camera feed
- ✅ MongoDB storage (image & metadata using GridFS)
- ✅ Email alert system (with cooldown)
- ✅ IP-based geolocation tagging
- ✅ Optional Vision Transformer (ViT/Swin) architecture integration



## 🧪 Tech Stack

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- OpenCV, PyTorch, NumPy
- MongoDB + GridFS
- SMTP (Gmail)
- Geocoder for location
- skimage for wavelet denoising
- Custom Vision Transformer for experimentation



## 🛠️ Setup Instructions


### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-surveillance.git
cd smart-surveillance
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 3. Download YOLOv8 Models

- General Model: [`yolov8n.pt`](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- Custom Drone Model: Place your `yolov8m-drone.pt` in the same directory.

> 📁 Ensure the weights are in the correct path or update the paths in the code accordingly.



## 🔐 Email Configuration

Edit the email credentials in the code:

```python
msg['From'] = 'your_email@gmail.com'
msg['To'] = 'recipient_email@gmail.com'
smtp.login('your_email@gmail.com', 'your_app_password')  # Use App Password
```

> ⚠️ Enable **2-Step Verification** and use **App Passwords** for Gmail.




## 🖥️ Run the Program

```bash
python main.py
```



## 📸 Screenshots
![image](https://github.com/user-attachments/assets/14bea68b-f905-4ecf-9766-6b04f60f3d5e)
![Screenshot 2025-02-19 225921](https://github.com/user-attachments/assets/9bcb84a3-15d7-48b1-a0c9-5b0ba4582e41)

![Screenshot (394)](https://github.com/user-attachments/assets/d29720a2-fa16-4e50-ac86-89c2514cb695)



## 📂 MongoDB Schema

- Database: `pjt2`
- Collections:
  - `human_detections`
  - `animal_detections`
  - `drone_detections`
- Image storage via GridFS
- Each record includes:
  - `timestamp`
  - `objects_detected`
  - `location`
  - `image_id`



## 🧑‍💻 Author

- **Aryan Singh**  
  *Built for academic/experimental use*



## 📜 License

This project is licensed under the MIT License.
