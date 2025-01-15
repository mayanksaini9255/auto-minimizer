import cv2
import time
import pyautogui
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import deque

# Model path (if the model is in the same directory as the script)
model = tf.saved_model.load('')  # Or provide the explicit path if different

# VERIFY CLASS ORDER - IMPORTANT! This must match your training data.
classes = ["unfocused", "focused"]

# CHANGE THIS SIZE IF NEEDED to match your model's input size
model_input_width = 300
model_input_height = 300

cap = cv2.VideoCapture(0)
previous_state = "focused"
minimized = False

# Parameters for smoothing and debouncing
buffer_size = 10  # Number of recent classifications to consider
state_buffer = deque(['focused'] * buffer_size, maxlen=buffer_size) # Initialize with 'focused'
unfocused_threshold = 0.7 # Percentage of 'unfocused' classifications to trigger minimization
unfocused_time_threshold = 3 # Seconds to remain unfocused before minimizing

start_unfocused_time = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_pil = Image.fromarray(frame)
    img_pil = img_pil.resize((model_input_width, model_input_height), Image.Resampling.LANCZOS)
    inp_numpy = np.array(img_pil)
    inp = tf.constant(inp_numpy, dtype='float32')
    inp = tf.expand_dims(inp, axis=0) # Add batch dimension

    prediction = model(inp)[0].numpy()
    predicted_class_index = np.argmax(prediction)
    current_state = classes[predicted_class_index]


    state_buffer.append(current_state)
    unfocused_ratio = state_buffer.count('unfocused') / buffer_size



    if unfocused_ratio >= unfocused_threshold:  # Check if unfocused for a sufficient duration
        if start_unfocused_time is None:
            start_unfocused_time = time.time()
        elif time.time() - start_unfocused_time >= unfocused_time_threshold and not minimized:
            pyautogui.hotkey('win', 'd')
            minimized = True
            print("Minimized windows")
            start_unfocused_time = None  # Reset the timer

    elif minimized: # If mostly focused and currently minimized
        pyautogui.hotkey('win', 'd')
        minimized = False
        print("Restored windows")
        start_unfocused_time = None  # Reset the timer



    cv2.putText(frame, current_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Focus Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break


cap.release()
cv2.destroyAllWindows()