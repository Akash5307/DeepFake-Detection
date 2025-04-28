import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import face_recognition
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Deepfake Image Detector',
    page_icon='🕵️‍♂️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Load model with caching
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def main():
    model = load_model()

    with st.sidebar:
        st.image('logo4.jpeg', use_column_width=True)
        st.title('Upload Your Image')
        img_uploaded = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        st.markdown(
            """
            ---
            📌 **Note:** This app uses a Vision Transformer model to detect deepfake images.
            """
        )

    st.title('🕵️‍♂️ DeepFake Image Detector')
    st.markdown(
        """
        Welcome to the DeepFake Detector app! Upload an image and find out whether it's **Real** or **Fake**.
        """
    )

    st.info('Please upload a clear image with a visible face.', icon="ℹ️")

    # Instructions Section
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader('How to Use:')
            st.markdown("""
            1. 📤 Upload an image from the sidebar.
            2. 🔍 The system will detect faces.
            3. 🧠 Analyze and predict Real or Fake.
            4. 💾 Download face crops if you want.
            """)
        # with col2:
            # st.image('https://media.giphy.com/media/26xBIygOcC3bAWg3O/giphy.gif', use_column_width=True)

    # Limitations section
    with st.expander('⚠️ Limitations', expanded=False):
        st.markdown('''
            - Variations in image quality, lighting conditions, and facial expressions may affect detection accuracy.
            - The model may not detect very advanced modern deepfakes.
            - It heavily depends on the diversity and quality of training data.
        ''')

    st.divider()

    # Image processing
    if img_uploaded is not None:
        with st.spinner('🛠️ Processing the image and detecting faces...'):
            image = cv2.imdecode(np.frombuffer(img_uploaded.read(), dtype=np.uint8), 1)
            face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            st.warning('No faces detected. Please upload a different image.', icon="⚠️")
        else:
            st.success(f'✅ {len(face_locations)} face(s) detected!', icon="✅")
        
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]

            with st.spinner(f'🔎 Preprocessing face #{i+1}...'):
                processed_face = preprocess_image(face_image)
                processed_face = np.expand_dims(processed_face, axis=0)

                prediction = model.predict(processed_face)
                predicted_class = "🛑 FAKE" if prediction[0, 0] > 0.5 else "✅ REAL"
                probability = prediction[0, 0]

                st.subheader(f'Face #{i+1} Result:')
                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB),
                         caption=f"Prediction: {predicted_class}\nConfidence: {probability:.2f}",
                         width=350)

                download_img = cv2.imencode('.png', face_image)[1].tobytes()
                st.download_button(
                    label="💾 Download Face Image",
                    data=download_img,
                    file_name=f"face_{i+1}_{predicted_class}.png",
                    mime="image/png"
                )

                if predicted_class == "🛑 FAKE":
                    st.error('Alert: This face appears to be a DeepFake!', icon="🚨")
                else:
                    st.success('Good news: This face appears Real!', icon="✅")

                st.divider()

if __name__ == "__main__":
    main()
