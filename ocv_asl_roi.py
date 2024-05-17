import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt
import sys


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

model = load_model('75epochtest.keras')  # Ensure this path is correct

def preprocess_image_for_prediction(img, target_size=(150, 150)):
    """
    Preprocess the input image for prediction.
    """
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign_language(img):
    """
    Predict the sign language gesture from the processed image.
    """
    processed_img = preprocess_image_for_prediction(img)
    predictions = model.predict(processed_img)
    pred_class = np.argmax(predictions, axis=1)[0]
    pred_certainty = np.max(predictions)
    return pred_class, pred_certainty

def detect_hands(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks :
        # Unhash for landmarks 
        #mp_drawing.draw_landmarks(image_rgb,results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr
    return img


def draw_landmarks(img,landmarks) :
    for i in range(len(landmarks.landmark)):
        h,w,c = img.shape
        cx, cy = int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)

        cv2.circle(img,(cx,cy), 10, (255,0,0), -1)

        if i < len(landmarks.landmark) - 1 :
            next_cx, next_cy = int(landmarks.landmark[i+1].x * w), int(landmarks.landmark[i+1].y * h)
            cv2.line(img,(cx,cy),(next_cx, next_cy), (0,255,0), 2)

def plot():

    history = model.history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def main():
    cam_capture = cv2.VideoCapture(0)
    if not cam_capture.isOpened():
        print("Error opening video stream or file")
        return

    x_start, y_start = 100, 100

    roi_width, roi_height = 400, 400

    while True:
        ret, frame = cam_capture.read()
        if not ret:
            break
        if ret :
            frame_detect = detect_hands(frame)


        



        cv2.rectangle(frame_detect, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (0, 255, 0), 2)
        roi = frame_detect[y_start:y_start + roi_height+100, x_start:x_start + roi_width+100]

        pred_class, pred_certainty = predict_sign_language(roi)


        label = chr(pred_class + 65)

        cv2.putText(frame_detect, f'Pred: {label}, Cert: {pred_certainty:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Sign Language Prediction', frame_detect)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_capture.release()
    cv2.destroyAllWindows()


if len(sys.argv) >= 2:
    if sys.argv[1] == "--plot":
        plot()
else:
    main()

# if __name__ == '__main__':
#     plot()
