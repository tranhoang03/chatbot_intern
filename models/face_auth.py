import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import sqlite3
from scipy.spatial.distance import cosine
import time
import json
import streamlit as st
import os
from pathlib import Path

# Get base directory
BASE_DIR = Path(__file__).parent.parent

def load_facenet_pb(model_path):
    # Use absolute path
    model_path = os.path.join(BASE_DIR, "models", model_path)
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def get_face_embedding(face_img, sess, input_tensor, embedding_tensor, phase_train_tensor):
    # Preprocess image
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    face_img = (face_img - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)
    
    # Get embedding
    embedding = sess.run(embedding_tensor, 
                        feed_dict={input_tensor: face_img, 
                                 phase_train_tensor: False})[0]
    return embedding

def find_matching_face(embedding, threshold=0.5):
    conn = sqlite3.connect('Database.db')
    cursor = conn.cursor()
    
    # Get all embeddings from database
    cursor.execute("SELECT embedding, name, id FROM customers")
    results = cursor.fetchall()
    
    best_match = None
    min_distance = float('inf')
    
    for db_embedding, name, id in results:
        try:
            # Convert string embedding back to numpy array
            db_embedding = np.array(json.loads(db_embedding), dtype=np.float32)
            
            # Calculate cosine distance
            distance = cosine(embedding, db_embedding)
            
            if distance < min_distance:
                min_distance = distance
                if distance < threshold:
                    best_match = {'name': name, 'id': id}
        except Exception as e:
            print(f"Error processing embedding for {name}: {e}")
            continue
    
    conn.close()
    return best_match

def capture_face():
    # Load YOLO model with absolute path
    yolo_model = YOLO(os.path.join(BASE_DIR, "models", "best.pt"))
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Biến để lưu kết quả
    captured_face = None
    start_time = None
    CAPTURE_DELAY = 5  # Thời gian chờ (giây)
    
    # Tạo placeholder cho camera feed
    camera_placeholder = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        
        # YOLO detection
        results = yolo_model(frame)
        
        # Process each detected face
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get face coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Bắt đầu đếm thời gian khi phát hiện khuôn mặt
                if start_time is None:
                    start_time = current_time
                
                # Xử lý khuôn mặt sau 5 giây
                if start_time is not None and current_time - start_time >= CAPTURE_DELAY:
                    try:
                        captured_face = frame[y1:y2, x1:x2].copy()
                        cap.release()
                        return captured_face
                        
                    except Exception as e:
                        print(f"Error capturing face: {e}")
                        cap.release()
                        return None
                
                # Hiển thị thời gian còn lại
                if start_time is not None:
                    remaining_time = CAPTURE_DELAY - (current_time - start_time)
                    label = f"Capturing in: {remaining_time:.1f}s"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update camera feed in Streamlit
        camera_placeholder.image(frame_rgb, channels="RGB")
        
        # Add a small delay to prevent overwhelming the UI
        time.sleep(0.1)
    
    # Release resources
    cap.release()
    return None

def process_face_recognition(face_img):
    # Use absolute path for FaceNet model
    model_path = os.path.join(BASE_DIR, "models", "20180402-114759.pb")
    facenet_graph = load_facenet_pb(model_path)
    
    # Get input and output tensors
    sess = tf.compat.v1.Session(graph=facenet_graph)
    input_tensor = facenet_graph.get_tensor_by_name("input:0")
    embedding_tensor = facenet_graph.get_tensor_by_name("embeddings:0")
    phase_train_tensor = facenet_graph.get_tensor_by_name("phase_train:0")
    
    try:
        # Get face embedding
        face_embedding = get_face_embedding(face_img, sess, input_tensor, 
                                         embedding_tensor, phase_train_tensor)
        
        # Find matching face in database
        match = find_matching_face(face_embedding)
        return match
            
    except Exception as e:
        print(f"Error processing face recognition: {e}")
        return None
    
    finally:
        sess.close()

def authenticate_user():
    st.markdown("### 👤 Vui lòng nhìn vào camera để xác thực")
    
    # Create a container for the camera
    camera_container = st.container()
    
    with camera_container:
        # Capture face
        face_img = capture_face()
    
    if face_img is not None:
        # Process face recognition
        user_info = process_face_recognition(face_img)
        
        if user_info:
            # Clear the camera container
            camera_container.empty()
            
            # Add a personalized greeting with animation
            st.markdown(f"""
            <div style='text-align: center; padding: 20px;'>
                <h2>👋 Xin chào {user_info['name']}!</h2>
                <p style='font-size: 1.2em;'>Rất vui được gặp lại bạn.</p>
            </div>
            """, unsafe_allow_html=True)
            
            return user_info
        else:
            st.error("Không tìm thấy thông tin người dùng trong hệ thống")
            return None
    else:
        st.error("Không thể chụp ảnh khuôn mặt")
        return None 