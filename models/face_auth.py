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
import av # Needed for VideoFrame
from streamlit_webrtc import VideoTransformerBase # Import base class
import queue # For communication

# Get base directory
BASE_DIR = Path(__file__).parent.parent

def load_facenet_pb(model_path):
    # Use absolute path
    model_path = os.path.join(BASE_DIR, "models", model_path)
    print(f"Loading FaceNet model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FaceNet model not found at: {model_path}")
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Create graph and import graph_def
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    print("FaceNet model loaded successfully.")
    return graph

def get_face_embedding(face_img, sess, input_tensor, embedding_tensor, phase_train_tensor):
    # Preprocess image
    try:
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32')
        # Normalize based on FaceNet requirements
        mean, std = face_img.mean(), face_img.std()
        face_img = (face_img - mean) / std # More robust normalization
        # face_img = (face_img - 127.5) / 128.0 # Original normalization
        face_img = np.expand_dims(face_img, axis=0)
        
        # Get embedding
        feed_dict = {input_tensor: face_img, phase_train_tensor: False}
        embedding = sess.run(embedding_tensor, feed_dict=feed_dict)[0]
        return embedding
    except Exception as e:
        print(f"Error during face preprocessing or embedding generation: {e}")
        raise # Re-raise the exception to be caught higher up

def find_matching_face(embedding, threshold=0.5):
    conn = sqlite3.connect('Database.db')
    cursor = conn.cursor()
    
    # Get all embeddings from database
    cursor.execute("SELECT embedding, name, id FROM customers")
    results = cursor.fetchall()
    
    best_match = None
    min_distance = float('inf')
    
    for db_embedding_str, name, id in results:
        try:
            db_embedding = np.array(json.loads(db_embedding_str), dtype=np.float32)
            distance = cosine(embedding, db_embedding)
            # print(f"Checking {name} (ID: {id}), Distance: {distance:.4f}") # Debug print
            if distance < min_distance:
                min_distance = distance
                if distance < threshold:
                    best_match = {'name': name, 'id': id}
        except Exception as e:
            print(f"Error processing DB embedding for {name}: {e}")
            continue
            
    # print(f"Best match: {best_match}, Min distance: {min_distance}") # Debug print
    conn.close()
    return best_match

class FaceAuthTransformer(VideoTransformerBase):
    def __init__(self, result_queue: queue.Queue, yolo_model_path="best.pt", facenet_model_path="20180402-114759.pb"):
        print("Initializing FaceAuthTransformer...") # Reverted print
        self.start_time = None
        self.capture_delay = 5  # seconds
        self.result_queue = result_queue
        self.match_found = False
        self.frame_count = 0 # Add frame counter for debugging

        # --- Load models ONCE --- 
        try:
            yolo_path = os.path.join(BASE_DIR, "models", yolo_model_path)
            print(f"Loading YOLO model from: {yolo_path}")
            if not os.path.exists(yolo_path):
                 raise FileNotFoundError(f"YOLO model not found at: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            print("YOLO model loaded successfully.")

            self.facenet_graph = load_facenet_pb(facenet_model_path)
            self.sess = tf.compat.v1.Session(graph=self.facenet_graph)
            self.input_tensor = self.facenet_graph.get_tensor_by_name("input:0")
            self.embedding_tensor = self.facenet_graph.get_tensor_by_name("embeddings:0")
            self.phase_train_tensor = self.facenet_graph.get_tensor_by_name("phase_train:0")
            print("FaceNet session and tensors initialized.")
            
        except Exception as e:
            print(f"!!! CRITICAL ERROR during model initialization: {e}")
            # Signal initialization failure (optional, depends on how you handle it in app)
            # self.result_queue.put(SystemError(f"Model init failed: {e}")) 
            raise # Reraise exception to potentially stop streamer creation
            
        print("FaceAuthTransformer initialized successfully.") # Reverted print

    # --- Internal method for face processing --- 
    def _process_face(self, face_img):
        try:
            face_embedding = get_face_embedding(face_img, self.sess, self.input_tensor, 
                                              self.embedding_tensor, self.phase_train_tensor)
            match = find_matching_face(face_embedding)
            return match
        except Exception as e:
            print(f"Error in _process_face: {e}")
            return None # Return None on error

    # --- Original recv method --- 
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        # print(f">>> recv called - Frame {self.frame_count}")

        if self.match_found:
             return frame # If match found, just return the frame without processing

        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        processed_img = img.copy() # Work on a copy to draw on

        try:
            # Run YOLO detection
            results = self.yolo_model(img, verbose=False)

            face_detected = False
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    face_detected = True
                    # Find the box with the highest confidence
                    best_box = max(boxes, key=lambda box: float(box.conf[0]))
                    x1, y1, x2, y2 = best_box.xyxy[0].int().tolist()
                    conf = float(best_box.conf[0])

                    # Draw bounding box
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put confidence score
                    cv2.putText(processed_img, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Crop the detected face
                    face_img = img[y1:y2, x1:x2]

                    # Check if face image is valid
                    if face_img.size == 0:
                        print(f"Warning: Empty face crop at Frame {self.frame_count}")
                        continue # Skip this problematic box

                    # Start timer only when a face is confidently detected
                    if self.start_time is None:
                        self.start_time = current_time

                    elapsed = current_time - self.start_time
                    remaining_time = max(0, self.capture_delay - elapsed)

                    # Display countdown or processing message
                    if elapsed >= self.capture_delay:
                        cv2.putText(processed_img, "Processing...", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        try:
                            # Perform face recognition
                            match = self._process_face(face_img)
                            user_info = match if match else None
                            print(f"Face recognition result: {user_info}") # Debug print
                            self.result_queue.put(user_info) # Send result (or None)
                            self.match_found = True # Stop processing after sending result
                            break # Exit loop once processed
                        except Exception as e:
                            print(f"!!! Error during face recognition (Frame {self.frame_count}): {e}")
                            self.result_queue.put(None) # Send None on error
                            self.match_found = True
                            break
                    else:
                         # Display remaining time
                         cv2.putText(processed_img, f"Wait: {remaining_time:.1f}s", (x1, y2 + 20),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    break # Only process the highest confidence face in the frame

            # If no face detected in this frame, reset the timer
            if not face_detected:
                self.start_time = None
                cv2.putText(processed_img, "No face detected", (50, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        except Exception as e:
            # Log error and return original frame maybe? Or processed with error message?
            print(f"!!! Error during YOLO inference or drawing (Frame {self.frame_count}): {e}")
            # Draw error message on the frame
            cv2.putText(processed_img, "Error processing frame", (50, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Decide whether to reset timer on error
            self.start_time = None

        # Return the processed frame with drawings
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

