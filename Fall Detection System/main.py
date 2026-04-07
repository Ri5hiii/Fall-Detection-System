import os
import cv2

base_path = r"D:\ML2.0\Video\fall_dataset\dataset\dataset"
output_folder = r"D:\frames"

os.makedirs(output_folder, exist_ok=True)

# 🔥 Better dataset (balanced + bigger)
train_chutes = [
    "chute01","chute02","chute03","chute04","chute07","chute08","chute09","chute10",
    "chute13","chute14","chute15","chute16","chute19","chute20","chute21","chute22"
]

val_chutes = [
    "chute05","chute06","chute11","chute12",
    "chute17","chute18","chute23","chute24"
]

for chute in train_chutes + val_chutes:
    chute_path = os.path.join(base_path, chute)

    label = "fall" if int(chute.replace("chute","")) >= 13 else "nonfall"
    split = "train" if chute in train_chutes else "val"

    for video in os.listdir(chute_path):
        if not video.endswith(".avi"):
            continue

        video_path = os.path.join(chute_path, video)
        cap = cv2.VideoCapture(video_path)

        save_path = os.path.join(output_folder, split, label, chute + "_" + video)
        os.makedirs(save_path, exist_ok=True)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 8 == 0:   # 🔥 more frames
                frame = cv2.resize(frame, (128,128))
                cv2.imwrite(f"{save_path}/frame_{saved_count}.jpg", frame)
                saved_count += 1

            if saved_count >= 25:  # 🔥 more data
                break

            frame_count += 1

        cap.release()

print("✅ Dataset Ready (Improved)")