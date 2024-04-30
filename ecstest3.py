import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks is not None:
        keypoints = np.zeros((len(results.pose_landmarks.landmark), 2))
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y]
        return keypoints
    else:
        return None
    
def process_images():
    poses = []
    labels = []
    for i in range(1, 2143):
        image_path = f"yogapose/Poses_renamed_aug/{str(i).zfill(3)}"
        
        image_formats = ["jpg", "png", "webp", "jpeg"]
        for ext in image_formats:
            full_path = f"{image_path}.{ext}"
            print("Trying to load:", full_path)
            image = cv2.imread(full_path)
            if image is not None:
                break
        
        if image is None:
            print("Error loading image:", image_path)
            continue
            
        keypoints = extract_pose(image)
        if keypoints is not None:
            poses.append(keypoints)
            
            if 1 <= i <= 91:
                labels.append(0)  # Anjaneyasana
            elif 92 <= i <= 168:
                labels.append(1)  # Adho Mukha Svasana
            elif 169 <= i <= 238:
                labels.append(2)  # Ardha Chakrasana
            elif 239 <= i <= 357:
                labels.append(3)  # Bhujangasana
            elif 358 <= i <= 427:
                labels.append(4)  # Chakrasana
            elif 428 <= i <= 574:
                labels.append(5)  # Dhanurasana
            elif 575 <= i <= 714:
                labels.append(6)  # Malasana
            elif 715 <= i <= 784:
                labels.append(7)  # Naukasana
            elif 785 <= i <= 931:
                labels.append(8)  # Paschimottasana
            elif 932 <= i <= 1036:
                labels.append(9)  # Shavasana
            elif 1037 <= i <= 1176:
                labels.append(10)  # Setu Bandha Sarvagasana
            elif 1177 <= i <= 1260:
                labels.append(11)  # Tadasana
            elif 1261 <= i <= 1414:
                labels.append(12)  # Trikonasana
            elif 1415 <= i <= 1519:
                labels.append(13)  # Uttanasana
            elif 1520 <= i <= 1596:
                labels.append(14)  # Ustrasana
            elif 1597 <= i <= 1666:
                labels.append(15)  # Utkatasana
            elif 1667 <= i <= 1813:
                labels.append(16)  # Vajrasana
            elif 1814 <= i <= 1883:
                labels.append(17)  # Virabhadrasan 1
            elif 1884 <= i <= 1953:
                labels.append(18)  # Virabhadrasan 2
            elif 1954 <= i <= 2023:
                labels.append(19)  # Virabhadrasan 3
            elif 2024 <= i <= 2142:
                labels.append(20)  # Vrikshasana
    return np.array(poses), np.array(labels)

poses, labels = process_images()

poses = poses / 640

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((33, 2, 1), input_shape=(33, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(poses, labels, epochs=100, batch_size=32)

model.save("yoga_pose_model")
