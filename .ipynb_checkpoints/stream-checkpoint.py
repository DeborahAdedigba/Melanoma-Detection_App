import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications import EfficientNetB4, InceptionResNetV2
import matplotlib.pyplot as plt

# Load skin image models
model_skin_cnn = load_model('OneDrive - Solent University/Diss/streamlit/Skin_models/CNN_skin_classifier_with_saliency.keras')
model_skin_vgg16 = load_model('OneDrive - Solent University/Diss/streamlit/Skin_models/best_VGG16_model.keras')
model_skin_resnet50 = load_model('OneDrive - Solent University/Diss/streamlit/Skin_models/best_ResNet50_model.keras')
model_skin_efficientnet = load_model('OneDrive - Solent University/Diss/streamlit/Skin_models/best_EfficientNetB4_model.keras')
model_skin_inceptionresnetv2 = load_model('OneDrive - Solent University/Diss/streamlit/Skin_models/best_InceptionResNetV2_model.keras')

# Load dermoscopy image models
model_derm_cnn = load_model('C:/Users/adedi/OneDrive - Solent University/Diss/streamlit/Demascopy_models/melanoma_cnn_with_saliency.h5')
model_derm_vgg16 = load_model('C:/Users/adedi/OneDrive - Solent University/Diss/streamlit/Demascopy_models/vgg16_model_with_saliency.h5')
model_derm_resnet50 = load_model('C:/Users/adedi/OneDrive - Solent University/Diss/streamlit/Demascopy_models/best_ResNet50_model.h5')
model_derm_efficientnet = load_model('C:/Users/adedi/OneDrive - Solent University/Diss/streamlit/Demascopy_models/efficientnetb4_model_with_saliency.h5')
model_derm_inceptionresnetv2 = load_model('C:/Users/adedi/OneDrive - Solent University/Diss/streamlit/Demascopy_models/InceptionResNetV2_model.h5')

# Disclaimer text
disclaimer_text = """
**Disclaimer:** This app is for educational purposes only. Consult a healthcare professional for accurate medical advice.
"""

# Main page - Introduction about Melanoma and the App
def main():
    st.title('Melanoma Detection App')
    st.markdown('''
        This app helps detect melanoma using AI models. Melanoma is a type of skin cancer.
        The app includes:
        - Information about melanoma
        - Model selection and performance
        - Visualizations
        - Detection using skin or dermoscopy images
        - Educational resources
        - FAQs
        - Feedback and contact
    ''')

# Model performance page
def model_performance_page():
    st.title('Model Performance')
    st.markdown('''
        ## Model Performance Metrics

        Choose a model from the sidebar to view its performance metrics.
    ''')

# Visualizations page
def visualize_data():
    st.title('Visualizations')
    st.markdown('''
        ## Visualizations

        Include visualizations related to melanoma detection or model performance.
    ''')

# Detection interface with model selection and image upload
def melanoma_detection():
    st.title('Melanoma Detection')

    # Model selection
    st.sidebar.title('Model Selection')
    model_type = st.sidebar.radio('Select Model Type', ['Skin Image Models', 'Dermoscopy Image Models'])

    if model_type == 'Skin Image Models':
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
    else:  # Dermscopy Image Models
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']

    selected_model = st.sidebar.selectbox('Select Model', models)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Process uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust target_size as per your model's input size

        if model_type == 'Skin Image Models':
            # Preprocess based on model
            if selected_model == 'VGG16':
                img = image.img_to_array(img)
                img = preprocess_vgg16(img)
            elif selected_model == 'ResNet50':
                img = image.img_to_array(img)
                img = preprocess_resnet50(img)
            elif selected_model == 'EfficientNetB4' or selected_model == 'InceptionResNetV2':
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)  # Assuming preprocess_input is defined for EfficientNetB4 and InceptionResNetV2
            else:  # Default for CNN model
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)  # CNN model expects 4D input (batch size, height, width, channels)

            # Model prediction
            if selected_model == 'CNN':
                prediction = model_skin_cnn.predict(img)
            elif selected_model == 'VGG16':
                prediction = model_skin_vgg16.predict(img)
            elif selected_model == 'ResNet50':
                prediction = model_skin_resnet50.predict(img)
            elif selected_model == 'EfficientNetB4':
                prediction = model_skin_efficientnet.predict(img)
            elif selected_model == 'InceptionResNetV2':
                prediction = model_skin_inceptionresnetv2.predict(img)

        else:  # Dermscopy Image Models
            # Preprocess based on model
            if selected_model == 'VGG16':
                img = image.img_to_array(img)
                img = preprocess_vgg16(img)
            elif selected_model == 'ResNet50':
                img = image.img_to_array(img)
                img = preprocess_resnet50(img)
            elif selected_model == 'EfficientNetB4' or selected_model == 'InceptionResNetV2':
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)  # Assuming preprocess_input is defined for EfficientNetB4 and InceptionResNetV2
            else:  # Default for CNN model
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)  # CNN model expects 4D input (batch size, height, width, channels)

            # Model prediction
            if selected_model == 'CNN':
                prediction = model_derm_cnn.predict(img)
            elif selected_model == 'VGG16':
                prediction = model_derm_vgg16.predict(img)
            elif selected_model == 'ResNet50':
                prediction = model_derm_resnet50.predict(img)
            elif selected_model == 'EfficientNetB4':
                prediction = model_derm_efficientnet.predict(img)
            elif selected_model == 'InceptionResNetV2':
                prediction = model_derm_inceptionresnetv2.predict(img)

        st.write(f"Prediction: {'Melanoma' if prediction > 0.5 else 'Not Melanoma'}")

    else:
        st.write("Please upload an image.")

# Educational resources section
def educational_resources():
    st.title('Educational Resources')
    st.markdown('''
        ## Educational Resources

        Learn more about Melanoma:
        - [American Cancer Society](https://www.cancer.org/cancer/melanoma-skin-cancer.html)
        - [Skin Cancer Foundation](https://www.skincancer.org/skin-cancer-information/melanoma/)
    ''')

# FAQs section
def faq_section():
    st.title('FAQs')
    st.markdown('''
        ## FAQs

        **Q: How accurate are the models used in this app?**
        - A: The accuracy varies per model. Check the Model Performance page for details.

        **Q: Can I trust the results of this app for medical diagnosis?**
        - A: This app is for educational purposes. Consult a healthcare professional for medical advice.

        **Q: What types of images can I upload for detection?**
        - A: You can upload skin images or dermoscopy images.

        **Q: How do I interpret the prediction results?**
        - A: Predictions are classified as either Melanoma or Not Melanoma with an explanation provided.

        **Q: Is my uploaded image stored or used for other purposes?**
        - A: No, uploaded images are only used for classification within the session and are not stored.
    ''')

# Feedback and contact form
def feedback_form():
    st.title('Feedback and Contact')
    st.write('Have feedback or need support? Contact us at your@email.com')

# Combine all pages into a single app
def run_app():
    st.sidebar.title('Navigation')
    pages = {
        "Introduction": main,
        "Model Performance": model_performance_page,
        "Visualizations": visualize_data,
        "Melanoma Detection": melanoma_detection,
        "Educational Resources": educational_resources,
        "FAQs": faq_section,
        "Feedback and Contact": feedback_form
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

    st.sidebar.markdown(disclaimer_text)

if __name__ == '__main__':
    run_app()
