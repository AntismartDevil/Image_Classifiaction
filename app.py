import streamlit as st
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from PIL import Image
except ModuleNotFoundError:
    st.error("TensorFlow is not installed. Please ensure TensorFlow is installed in the environment.")


st.set_page_config(page_title="Image Classifier", page_icon=":camera:", layout="wide")


@st.cache_resource
def load_cifar10_model():
    try:
        return load_model('image_classification_model.h5')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_cifar10_model()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


st.markdown(
    """
    <style>
    body {
        background-color: #141414; 
        color: #e5e5e5; 
    }
    .main {
        background-color: #141414; 
        color: #e5e5e5;
        padding: 20px;
    }
    h1 {
        color: #e5e5e5; 
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
        font-size: 3rem;
    }
    h2, h3, h4, h5, h6, p {
        color: #e5e5e5; 
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #e50914; 
        color: white;
        border-radius: 5px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        margin-top: 20px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #b81d24; 
    }
    .stFileUploader {
        background-color: #333333;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
    }
    .stImage {
        max-width: 100%; 
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000000;
        color: #e5e5e5;
        text-align: center;
        padding: 10px;
    }
    .content-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)


st.title('Image Classifier :camera:')


st.write('Upload an image and let the model predict which class it belongs to!')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=False, width=300)
    
    with st.spinner('Processing...'):
        try:
           
            img = Image.open(uploaded_file)
            img = img.resize((32, 32))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

          
            if model:
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction[0])
                class_name = class_names[predicted_class]
                st.write(f"### Prediction: {class_name.capitalize()}")
                st.balloons()
            else:
                st.error("Model could not be loaded. Please check your environment or model file.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


st.subheader("Feedback")
feedback = st.text_area("Leave your feedback or comments here:")

if st.button('Submit Feedback'):
    if feedback:
        st.write("Thank you for your feedback!")
        st.write(f"**Your Feedback:** {feedback}")
    else:
        st.warning("Please enter some feedback before submitting.")


st.markdown(
    """
    <div class="footer">
        <p>Developed by <b>Aditya Wagh</b> Under Guidance of <b>Prof. Om Prakash</b></p>
    </div>
    """, unsafe_allow_html=True
)
