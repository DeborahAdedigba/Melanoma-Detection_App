import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
import matplotlib.pyplot as plt


# Load skin image models
try:
    model_skin_cnn = load_model('CNN_skin_classifier_with_saliency.keras')
    model_skin_vgg16 = load_model('best_VGG16_model.keras')
    model_skin_resnet50 = load_model('best_ResNet50_model.keras')
    model_skin_efficientnet = load_model('best_EfficientNetB4_model.keras')
    model_skin_inceptionresnetv2 = load_model('best_InceptionResNetV2_model.keras')
except Exception as e:
    st.error(f"Error loading skin image models: {e}")

# Load dermoscopy image models
try:
    model_derm_cnn = load_model('DM_melanoma_cnn_with_saliency.keras')
    model_derm_vgg16 = load_model('DM_vgg16_model_with_saliency.keras')
    model_derm_resnet50 = load_model('DM_best_ResNet50_model.keras')
    model_derm_efficientnet = load_model('DM_efficientnetb4_model_with_saliency.keras')
    model_derm_inceptionresnetv2 = load_model('DM_InceptionResNetV2_model.keras')
except Exception as e:
    st.error(f"Error loading dermoscopy image models: {e}")

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

    model_type = st.sidebar.selectbox('Select Model Type', ['Skin Image Models', 'Dermoscopy Image Models'])

    if model_type == 'Skin Image Models':
        models = {
            'CNN': model_skin_cnn,
            'VGG16': model_skin_vgg16,
            'ResNet50': model_skin_resnet50,
            'EfficientNetB4': model_skin_efficientnet,
            'InceptionResNetV2': model_skin_inceptionresnetv2
        }
    else:  # Dermoscopy Image Models
        models = {
            'CNN': model_derm_cnn,
            'VGG16': model_derm_vgg16,
            'ResNet50': model_derm_resnet50,
            'EfficientNetB4': model_derm_efficientnet,
            'InceptionResNetV2': model_derm_inceptionresnetv2
        }

    selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))
    model = models[selected_model]

    # Display model summary or metrics
    st.subheader(f"{selected_model} Model Performance")
    st.text(model.summary())

    # Add more performance metrics here (accuracy, loss, confusion matrix, etc.)

# Visualizations page
def visualize_data():
    st.title('Visualizations')
    st.markdown('''
        ## Visualizations

        Include visualizations related to melanoma detection or model performance.
    ''')
# dection 

import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image

import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image

def melanoma_detection():
    st.title('Melanoma Detection')

    # Model selection
    st.sidebar.title('Model Selection')
    model_type = st.sidebar.radio('Select Model Type', ['Skin Image Models', 'Dermoscopy Image Models'])

    if model_type == 'Skin Image Models':
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
        model_dict = {
            'CNN': model_skin_cnn,
            'VGG16': model_skin_vgg16,
            'ResNet50': model_skin_resnet50,
            'EfficientNetB4': model_skin_efficientnet,
            'InceptionResNetV2': model_skin_inceptionresnetv2
        }
        preprocess_dict = {
            'VGG16': preprocess_vgg16,
            'ResNet50': preprocess_resnet50,
            'EfficientNetB4': preprocess_efficientnet,
            'InceptionResNetV2': preprocess_inceptionresnetv2
        }
    else:  # Dermoscopy Image Models
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
        model_dict = {
            'CNN': model_derm_cnn,
            'VGG16': model_derm_vgg16,
            'ResNet50': model_derm_resnet50,
            'EfficientNetB4': model_derm_efficientnet,
            'InceptionResNetV2': model_derm_inceptionresnetv2
        }
        preprocess_dict = {
            'VGG16': preprocess_vgg16,
            'ResNet50': preprocess_resnet50,
            'EfficientNetB4': preprocess_efficientnet,
            'InceptionResNetV2': preprocess_inceptionresnetv2
        }

    selected_model = st.sidebar.selectbox('Select Model', models)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        try:
            # Determine input size based on the selected model
            if selected_model == 'EfficientNetB4':
                input_size = (380, 380)
            elif selected_model == 'InceptionResNetV2':
                input_size = (299, 299)
            else:
                input_size = (224, 224)

            # Process uploaded image
            img = image.load_img(uploaded_file, target_size=input_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # Preprocess based on model
            if selected_model in preprocess_dict:
                img = preprocess_dict[selected_model](img)

            # Assuming saliency map is generated (dummy saliency map for illustration)
            saliency_map = np.zeros_like(img)

            # Concatenate the original image with the saliency map
            combined_input = np.concatenate((img, saliency_map), axis=-1)

            # Model prediction
            model = model_dict[selected_model]
            prediction = model.predict(combined_input)

            # Handle the prediction output
            if prediction.shape[-1] > 1:  # If the model outputs multiple classes
                predicted_class = np.argmax(prediction, axis=-1)
                confidence = np.max(prediction)
            else:  # If the model outputs a single value
                threshold = 0.5  # Adjust this threshold as needed
                predicted_class = (prediction > threshold).astype(int)
                confidence = prediction[0][0]

            # Output based on model type
            if model_type == 'Skin Image Models':
                result = 'Melanoma' if predicted_class[0] == 1 else 'Non-Melanoma'
            else:  # Dermoscopy Image Models
                result = 'Malignant' if predicted_class[0] == 1 else 'Benign'

            st.write(f"Prediction: {result}")
            st.write(f"Confidence: {confidence:.2f}")

            # Additional information
            st.write("\nPlease note:")
            st.write("- This prediction is based on the model's analysis and should not be considered as a definitive medical diagnosis.")
            st.write("- If you have any concerns about a skin lesion, please consult with a qualified healthcare professional or dermatologist.")
            st.write("- Regular skin check-ups and early detection are crucial for managing melanoma risk.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
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
