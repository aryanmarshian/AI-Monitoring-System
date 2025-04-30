import cv2
import numpy as np
import datetime
import os
import smtplib
import pymongo
import gridfs
import geocoder
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from ultralytics import YOLO
from skimage.restoration import denoise_wavelet
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# === MongoDB Connection ===
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "pjt2"
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
fs = gridfs.GridFS(db)  # For image storage

# === Load AI Models ===
print("Loading models...")
model_general = YOLO("yolov8n.pt")  # General Model (Humans, Animals)

# Drone model - ensure you have the correct path to your drone model
# You might need to adjust this path
try:
    drone_model = YOLO("yolov8m-drone.pt")  # Dedicated Drone Model
    print("Drone model loaded successfully!")
except Exception as e:
    print(f"Error loading drone model: {e}")
    print("Attempting to use general model for all detections...")
    drone_model = None

print("Models loaded successfully!")

# Print class names from each model for debugging
print("\nGeneral model classes:")
for idx, class_name in model_general.names.items():
    print(f"Class {idx}: {class_name}")

if drone_model and hasattr(drone_model, 'names'):
    print("\nDrone model classes:")
    for idx, class_name in drone_model.names.items():
        print(f"Class {idx}: {class_name}")

# === Image Enhancement Function ===
def enhance_image(image):
    """Apply denoising and contrast enhancement for better detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply wavelet denoising
    denoised = denoise_wavelet(gray)
    
    # Convert back to uint8 and enhance contrast
    denoised_uint8 = (denoised * 255).astype(np.uint8)
    enhanced = cv2.equalizeHist(denoised_uint8)
    
    # Convert back to BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# === Get Camera Location ===
def get_camera_location():
    try:
        g = geocoder.ip('me')
        if g.latlng:
            return f"Lat: {g.latlng[0]}, Lon: {g.latlng[1]}"
        return "Location unavailable"
    except Exception as e:
        return f"Location error: {e}"

# === Save to MongoDB ===
def save_to_mongodb(timestamp, detected_objects, location_info, category, image):
    """Stores detection results and images in MongoDB."""
    _, buffer = cv2.imencode(".jpg", image)
    image_id = fs.put(buffer.tobytes(), filename=f"{timestamp}.jpg")
    
    collection = db[category.lower() + "_detections"]
    collection.insert_one({
        "timestamp": timestamp,
        "objects_detected": detected_objects,
        "location": location_info,
        "image_id": image_id
    })
    print(f"Saved {category} detection to MongoDB")

# === Send Email Alert ===
import time

# Initialize a global variable to track the last email time
last_email_time = 0  # seconds since epoch

def send_email_with_image(image, detected_objects):
    """Sends an alert email when an object is detected, with a 30-second cooldown."""
    global last_email_time
    current_time = time.time()

    # Check if 30 seconds have passed since last email
    if current_time - last_email_time < 5:
        print("Warm-up: Email not sent, please wait before sending another.")
        return False

    try:
        msg = MIMEMultipart()
        msg['Subject'] = '🚨 High Confidence Object Detected'
        msg['From'] = 'ironsingh2003@gmail.com'
        msg['To'] = 'aryansingh9503@gmail.com'
        
        # Add text
        text = f"Alert! Detected: {', '.join(detected_objects)}"
        msg.attach(MIMEText(text))
        
        # Attach the image
        _, img_encoded = cv2.imencode('.jpg', image)
        msg.attach(MIMEImage(img_encoded.tobytes(), name="detection.jpg"))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login('ironsingh2003@gmail.com', 'zwwa puzb jmbb xkmb')
            smtp.send_message(msg)
            print("Email sent successfully!")
            last_email_time = current_time  # Update the last email time
            return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def get_vision_transformer(model_type='vit', img_size=224, patch_size=16, window_size=7,
                           in_chans=3, num_classes=10, embed_dim=96, depth=2, num_heads=4):

    class MLP(nn.Module):
        def __init__(self, dim, mlp_ratio=4):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * mlp_ratio)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * mlp_ratio, dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class WindowAttention(nn.Module):
        def __init__(self, dim, window_size, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, is_swin=False, input_size=None):
            super().__init__()
            self.is_swin = is_swin
            self.norm1 = nn.LayerNorm(dim)
            self.attn = WindowAttention(dim, window_size, num_heads)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP(dim)

            if is_swin:
                self.H, self.W = input_size

        def forward(self, x):
            if self.is_swin:
                B, N, C = x.shape
                x = x.view(B, self.H, self.W, C)
                x = rearrange(x, 'b (h w1) (w w2) c -> (b h w) (w1 w2) c', w1=window_size, w2=window_size)
                x = self.attn(self.norm1(x)) + x
                x = self.mlp(self.norm2(x)) + x
                x = rearrange(x, '(b h w) (w1 w2) c -> b (h w1) (w w2) c',
                              b=B, h=self.H//window_size, w=self.W//window_size, w1=window_size, w2=window_size)
                return x.view(B, -1, C)
            else:
                x = self.attn(self.norm1(x)) + x
                x = self.mlp(self.norm2(x)) + x
                return x

    class VisionTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            num_patches = (img_size // patch_size) ** 2
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if model_type == 'vit' else None
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + (1 if self.cls_token is not None else 0), embed_dim))
            self.blocks = nn.Sequential(*[
                TransformerBlock(embed_dim, num_heads, is_swin=(model_type == 'swin'),
                                 input_size=(img_size // patch_size, img_size // patch_size))
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, C)
            B, N, C = x.shape

            if self.cls_token is not None:
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls, x), dim=1)

            x = x + self.pos_embed[:, :x.size(1)]
            x = self.blocks(x)
            x = self.norm(x)

            return self.head(x[:, 0]) if self.cls_token is not None else self.head(x.mean(dim=1))

    return VisionTransformer()


# === Process Detections from General Model ===
def process_general_detections(results, frame):
    """Process detections from the general YOLO model (humans, animals)"""
    detected_objects = []
    result_frame = frame.copy()
    
    # Process each detection
    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
        class_id = int(class_id)
        
        if conf < 0.7:  # Confidence threshold
            continue
        
        # Get category name from model
        category = results[0].names[class_id]
        
        # Determine object type based on class ID and name
        if class_id == 0:  # Person in COCO dataset
            detection_type = "Human"
            color = (0, 255, 0)  # Green for humans
        elif 15 <= class_id <= 25:  # Animals range in COCO dataset
            detection_type = "Animal"
            color = (255, 255, 0)  # Yellow for animals
        else:
            continue  # Skip other objects
        
        detected_objects.append(f"{detection_type} ({conf:.2f})")
        
        # Draw bounding box
        cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(result_frame, f"{detection_type} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return detected_objects, result_frame

# === Process Detections from Drone Model ===
def process_drone_detections(results, frame):
    """Process detections from the drone-specific model"""
    detected_objects = []
    result_frame = frame.copy()
    
    # Lower confidence threshold for drone detection
    confidence_threshold = 0.3  # Lower threshold for drone detection
    
    # Process each detection
    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
        class_id = int(class_id)
        
        if conf < confidence_threshold:
            continue
        
        # For custom drone model:
        # 1. Use all detected classes as potential drones
        # 2. Check class names if they contain 'drone'
        
        if class_id in results[0].names:
            category = results[0].names[class_id]
            # Check if this is a drone class (either by name or assuming all detections are drones)
            detection_type = "Drone"
            
            color = (0, 0, 255)  # Red for drones
            detected_objects.append(f"{detection_type} ({conf:.2f})")
            
            # Draw bounding box
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(result_frame, f"{detection_type} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return detected_objects, result_frame

# === Use General Model for Drone Detection if needed ===
def detect_drones_with_general_model(frame):
    """Fallback to detect drones using the general model"""
    detected_objects = []
    result_frame = frame.copy()
    
    # Run detection with general model
    results = model_general(frame)
    
    # Process each detection, looking for objects that might be drones
    for det in results[0].boxes.data:
        x1, y1, x2, y2, conf, class_id = det.cpu().numpy()
        class_id = int(class_id)
        
        # Lower confidence for potential drones
        if conf < 0.4:
            continue
        
        # Class 4 is airplane in COCO, might help detect drones
        # Also checking for objects in the air (birds, kites, etc.)
        drone_related_classes = [4, 5, 6, 7, 38]  # airplane, bus, train, truck, kite
        
        if class_id in drone_related_classes:
            category = results[0].names[class_id]
            detection_type = f"Potential Drone ({category})"
            
            color = (0, 0, 255)  # Red for drones
            detected_objects.append(f"{detection_type} ({conf:.2f})")
            
            # Draw bounding box
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(result_frame, f"{detection_type}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return detected_objects, result_frame

# === Main Detection Logic ===
def detect_objects(frame):
    """Process a single frame with both models and combine results"""
    # Make copies of the frame for each model
    general_frame = frame.copy()
    
    # Run general model detection (humans, animals)
    general_results = model_general(general_frame)
    general_detections, general_frame = process_general_detections(general_results, general_frame)
    
    # Initialize drone variables
    drone_detections = []
    drone_frame = frame.copy()
    
    # Try to use the dedicated drone model if available
    if drone_model:
        drone_results = drone_model(drone_frame)
        drone_detections, drone_frame = process_drone_detections(drone_results, drone_frame)
    
    # If no drone detections and no dedicated model, try using general model
    if not drone_detections:
        fallback_drone_detections, fallback_drone_frame = detect_drones_with_general_model(frame)
        if fallback_drone_detections:
            drone_detections = fallback_drone_detections
            drone_frame = fallback_drone_frame
    
    # Combine results onto original frame
    combined_frame = frame.copy()
    
    # If human/animal detections, copy to combined frame
    if general_detections:
        combined_frame = general_frame.copy()
    
    # Add drone detections to combined frame
    if drone_detections:
        # We need to redraw drone detections on combined frame
        # For simplicity, just overlay the drone frame with some transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(drone_frame, alpha, combined_frame, 1 - alpha, 0, combined_frame)
    
    # Combine all detections for reporting
    all_detections = general_detections + drone_detections
    
    # Handle MongoDB storage and alerts if detections exist
    if all_detections:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        location_info = get_camera_location()
        
        # Store different types separately in MongoDB
        if general_detections:
            for detection in general_detections:
                if "Human" in detection:
                    save_to_mongodb(timestamp, [detection], location_info, "Human", combined_frame)
                elif "Animal" in detection:
                    save_to_mongodb(timestamp, [detection], location_info, "Animal", combined_frame)
        
        if drone_detections:
            save_to_mongodb(timestamp, drone_detections, location_info, "Drone", combined_frame)
        
        # Send email alert with the combined image
        send_email_with_image(combined_frame, all_detections)
    
    return combined_frame, all_detections

# === Main Video Processing Function ===
def main():
    
    print("Starting webcam capture...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # CAP_DSHOW often improves performance on Windows
    # Set a fixed resolution (e.g., 640x480 for smoother performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Optional: Try setting FPS to reduce strain (30 is generally a good balance)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 
    
    print("Webcam opened successfully at 640x480. Press 'q' to quit.")
   
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame. Exiting...")
            break
        
        # Process frame
        processed_frame, detections = detect_objects(frame)
        
        # If objects detected, show them in console
        if detections:
            print(f"Detected: {detections}")
        
        # Add text to show what's being detected
        cv2.putText(processed_frame, f"Detections: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the result
        cv2.imshow('Object Detection', processed_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()