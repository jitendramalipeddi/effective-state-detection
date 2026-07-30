import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download
import torch

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # Download weights directly from Hugging Face Hub at runtime
    discrete_path = hf_hub_download(repo_id="your-username/affective-emotion-weights", filename="model_discrete.pth")
    va_path = hf_hub_download(repo_id="your-username/affective-emotion-weights", filename="model_va.pth")
    
    # Load state dicts from downloaded temp paths
    model_discrete.load_state_dict(torch.load(discrete_path, map_location=device))
    model_va.load_state_dict(torch.load(va_path, map_location=device))
    
    return model_discrete, model_va, mtcnn, emotions, device

st.set_page_config(page_title="Affective State Detector", layout="wide")
st.title("Real-Time Affective State & Emotion Detector")
st.write("Allow camera access to analyze discrete emotions and dimensional affective states (Valence/Arousal) in real-time.")

# 1. CACHE THE MODELS
# The @st.cache_resource decorator ensures models are loaded only once and kept in memory
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    emotions = ['Angry', 'Frustrated', 'Sad', 'Sleepy', 'Surprise']
    
    # Load Discrete Model
    model_discrete = models.efficientnet_b0(weights=None)
    model_discrete.classifier[1] = nn.Linear(model_discrete.classifier[1].in_features, len(emotions))
    model_discrete.load_state_dict(torch.load("model_discrete.pth", map_location=device))
    model_discrete.eval()
    
    # Load VA Model
    model_va = models.resnet18(weights=None)
    model_va.fc = nn.Linear(model_va.fc.in_features, 2)
    model_va.load_state_dict(torch.load("model_va.pth", map_location=device))
    model_va.eval()
    
    # Load MTCNN
    mtcnn = MTCNN(image_size=224, margin=20, device=device, post_process=False)
    
    return model_discrete, model_va, mtcnn, emotions, device

model_discrete, model_va, mtcnn, emotions, device = load_models()
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 2. DEFINE THE WEBRTC CALLBACK
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert WebRTC frame to numpy array for OpenCV
    img = frame.to_ndarray(format="bgr24") 
    
    # Convert to RGB for MTCNN and PIL
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # Detect Face (get bounding boxes to draw on the screen)
    boxes, _ = mtcnn.detect(pil_img)
    
    if boxes is not None:
        # Get the cropped tensor for the model
        face_tensor = mtcnn(pil_img)
        
        if face_tensor is not None:
            # Prepare tensor for inference
            face_input = face_tensor.unsqueeze(0).to(device) / 255.0
            face_input = norm(face_input[0]).unsqueeze(0)
            
            with torch.no_grad():
                out_disc = model_discrete(face_input)
                pred_idx = torch.argmax(out_disc, dim=1).item()
                predicted_emotion = emotions[pred_idx]
                
                out_va = model_va(face_input)
                v, a = out_va[0][0].item(), out_va[0][1].item()
            
            # Draw bounding box and text on the OpenCV image
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display metrics above the bounding box
                text = f"{predicted_emotion} | V: {v:.2f} A: {a:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    # Return the processed frame to the web browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. CONFIGURE STUN SERVERS (Crucial for cloud hosting)
# This helps establish the media connection when hosted on Streamlit Cloud
rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# 4. LAUNCH THE STREAM
webrtc_streamer(
    key="emotion-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
