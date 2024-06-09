import streamlit as st
import numpy as np  
import tensorflow as tf
import cv2
import face_recognition
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Deepfake Detector',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='auto',
)

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('models/model0505.h5')
    return model

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def main():
    model = load_model()

    with st.sidebar:
        st.sidebar.image('media/logo.jpeg', use_column_width=True)
        with st.container():
            img_uploaded = st.file_uploader(
                "Upload an image...", type=["jpg", "png"])

        st.info(
            'A deep learning based model for detecting deepfake images.')
         
    st.title('DeepFake Detector')
    st.markdown('''
        A demo app for detecting deepfake images.
    ''')
    colA, colB = st.columns(2)
    colA.markdown('''
        **Steps to use this app:**
        1. Upload an image.
        2. Image will be processed.
        3. Real/Fake images will be identified.            
    ''')
    colB.markdown('''
    ''')

    st.caption('**Note:** Kindly read about limitations before using the application.')
    with st.expander('Limitations', expanded=False):
        st.markdown('''
            1. Variations in image quality, lighting conditions, and range of face expressions could all affect how well the model performs.

            2. The model might not be fully robust against emerging deepfake generation techniques. Modern image generation methods use advanced descriptors to assess the quality of the image's realism, so photos of such faces are difficult to distinguish from real people. 

            3. The model's performance heavily relies on the quality and diversity of the dataset used for training.
        ''')

    if img_uploaded is not None:
        with st.spinner('Processing the image, getting faces...'):
            image = cv2.imdecode(np.frombuffer(
                img_uploaded.read(), dtype=np.uint8), 1)
            face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            st.warning('Faces not found!')
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            with st.spinner(f'Preprocessing the face #{i+1}:'):
                processed_face = preprocess_image(face_image)
                processed_face = np.expand_dims(processed_face, axis=0)

                prediction = model.predict(processed_face)

                predicted_class = "FAKE" if prediction[0,
                                                       0] > 0.5 else "REAL"

                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB),
                         caption=f"Face {i+1}: {predicted_class} | Probability of image being fake: {prediction[0, 0]:.2f}", width=350)

                download_img = cv2.imencode('.png', face_image)[1].tobytes()
                st.download_button(label="Download Image", data=download_img,
                                   file_name=f"face_{i+1}_{predicted_class}.png", mime="image/png")

                if predicted_class == "FAKE":
                    st.warning('The image is fake!',  icon="⚠️")
                else:
                    st.success('The image is real!', icon="✅")

                st.divider()


if __name__ == "__main__":
    main()
