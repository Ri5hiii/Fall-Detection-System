# 📌 Project 4: Fall Detection System (Video Classification)

## 🔹 Description
This project focuses on automated fall detection using deep learning and computer vision techniques for healthcare monitoring and intelligent surveillance systems. The system analyzes video input and classifies human activities into two categories: **Fall and Non-Fall**, enabling timely detection and response in critical situations.

The project uses a multi-camera video dataset where frames are extracted and processed to capture spatial features for classification. Multiple deep learning models are implemented and compared to evaluate performance.

---

## 🔹 Models Used
- MobileNetV2-based CNN (Transfer Learning)
- Custom CNN (Baseline Model)
- EfficientNetB0-based CNN (Advanced Transfer Learning)

---

## 🔹 Key Features
- 🎥 Video-based human activity recognition  
- 🧠 Deep learning models for fall detection  
- ⚡ Transfer learning for improved accuracy  
- 📊 Performance evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC  
- 🖥️ Streamlit-based UI for real-time video analysis  
- 📈 Visualization of predictions and frame distribution  

---

## 🔹 Dataset
- Multi-camera fall detection dataset (Kaggle)  
- 24 video sequences (12 fall, 12 non-fall)  
- 8 camera views per sequence  
- ~50,000+ frames extracted  
- Preprocessed to 224×224 resolution  

---

## 🔹 Model Performance
- EfficientNetB0 (Best Model): ~91.3% accuracy  
- MobileNetV2: ~89.2% accuracy  
- Custom CNN: ~78.5% accuracy  

👉 EfficientNetB0 provides highest accuracy, while MobileNetV2 offers a balance between performance and efficiency.

---

## 🔹 Working Flow
1. Upload video input  
2. Extract frames from video  
3. Preprocess frames (resize + normalization)  
4. Feature extraction using CNN / Transfer Learning  
5. Classification (Fall / Non-Fall)  
6. Display prediction with confidence and graphs  

---

## 🔹 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Streamlit  

---

## 🔹 Conclusion
This project demonstrates the effectiveness of deep learning models in video-based fall detection systems. Transfer learning models significantly outperform traditional CNN approaches. EfficientNetB0 achieves the highest accuracy, while MobileNetV2 provides a lightweight and efficient solution for real-time applications.
