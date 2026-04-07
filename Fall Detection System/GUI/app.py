import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tempfile

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🧠 Fall Detection System", layout="wide")

# =========================
# BACKGROUND (WORKING)
# =========================
st.markdown("""
<style>
.stApp {
        background-image: url("https://images.unsplash.com/photo-1507149833265-60c372daea22");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<h1 style='text-align:center;color:white;'>🧠 Fall Detection System</h1>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model_A.h5")

# =========================
# SELECT INPUT
# =========================
option = st.radio("Choose Input Type", ["Image", "Video"])

# =========================
# IMAGE SECTION
# =========================
if option == "Image":

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)

        # preprocess
        img = np.array(image)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]
        confidence = float(pred)

        # =========================
        # RESULT CARD
        # =========================
        with col2:
            st.markdown("## 🎯 Prediction Result")

            if confidence > 0.5:
                st.error(f"⚠️ FALL DETECTED\n\nConfidence: {confidence*100:.2f}%")
            else:
                st.success(f"✅ NON-FALL\n\nConfidence: {(1-confidence)*100:.2f}%")

            # =========================
            # PROGRESS BAR
            # =========================
            st.markdown("### 🔋 Confidence Level")
            st.progress(int(confidence * 100))

            # =========================
            # BAR GRAPH
            # =========================
            st.markdown("### 📊 Probability Distribution")

            labels = ["Non-Fall", "Fall"]
            values = [1-confidence, confidence]

            fig = plt.figure()
            plt.bar(labels, values)
            plt.title("Prediction Probability")

            st.pyplot(fig)

            # =========================
            # PIE CHART
            # =========================
            st.markdown("### 🥧 Class Distribution")

            fig2 = plt.figure()
            plt.pie(values, labels=labels, autopct='%1.1f%%')
            st.pyplot(fig2)

# =========================
# VIDEO SECTION
# =========================
elif option == "Video":

    video_file = st.file_uploader("📤 Upload Video", type=["mp4","avi","mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        fall_count = 0
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (128,128))
            img = frame_resized / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]

            total_frames += 1
            if pred > 0.5:
                fall_count += 1

            stframe.image(frame, channels="BGR")

        cap.release()

        # =========================
        # FINAL DASHBOARD
        # =========================
        st.markdown("## 🎯 Video Analysis Dashboard")

        fall_percentage = (fall_count / total_frames) * 100
        safe_percentage = 100 - fall_percentage

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Frames", total_frames)
        col2.metric("Fall Frames", fall_count)
        col3.metric("Fall %", f"{fall_percentage:.2f}%")

        # FINAL RESULT
        if fall_percentage > 30:
            st.error("⚠️ FALL DETECTED IN VIDEO")
        else:
            st.success("✅ SAFE VIDEO")

        # =========================
        # SMALL GRAPHS (SIDE BY SIDE)
        # =========================
        colA, colB = st.columns(2)

        with colA:
            fig = plt.figure(figsize=(4,3))
            plt.bar(["Fall", "Safe"], [fall_count, total_frames - fall_count])
            plt.title("Frame Analysis", fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(fig, use_container_width=False)

        with colB:
            fig2 = plt.figure(figsize=(4,3))
            plt.pie(
                [fall_percentage, safe_percentage],
                labels=["Fall", "Safe"],
                autopct='%1.1f%%',
                textprops={'fontsize': 8}
            )
            plt.title("Distribution", fontsize=10)
            st.pyplot(fig2, use_container_width=False)

# =========================
# FOOTER
# =========================
st.markdown("<h4 style='text-align:center;color:white;'>💡 Powered by Deep Learning</h4>", unsafe_allow_html=True)