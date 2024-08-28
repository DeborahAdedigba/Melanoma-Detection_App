# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
# from tensorflow.keras.regularizers import l2

# def build_model(input_shape=(224, 224, 6)):  # 6 channels for image + saliency
#     inputs = Input(input_shape)
#     image_input = inputs[:, :, :, :3]
#     saliency_input = inputs[:, :, :, 3:]

#     def create_cnn(input_tensor):
#         conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
#         conv1 = BatchNormalization()(conv1)
#         conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv1)
#         conv1 = BatchNormalization()(conv1)
#         pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#         pool1 = Dropout(0.5)(pool1)
#         conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool1)
#         conv2 = BatchNormalization()(conv2)
#         conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv2)
#         conv2 = BatchNormalization()(conv2)
#         pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#         pool2 = Dropout(0.5)(pool2)
#         conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool2)
#         conv3 = BatchNormalization()(conv3)
#         conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv3)
#         conv3 = BatchNormalization()(conv3)
#         pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#         pool3 = Dropout(0.5)(pool3)
#         conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool3)
#         conv4 = BatchNormalization()(conv4)
#         conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv4)
#         conv4 = BatchNormalization()(conv4)
#         pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#         pool4 = Dropout(0.5)(pool4)
#         conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool4)
#         conv5 = BatchNormalization()(conv5)
#         conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv5)
#         conv5 = BatchNormalization()(conv5)
#         pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
#         pool5 = Dropout(0.5)(pool5)
#         flatten = Flatten()(pool5)
#         return flatten

#     image_features = create_cnn(image_input)
#     saliency_features = create_cnn(saliency_input)
#     concatenated_features = tf.keras.layers.Concatenate()([image_features, saliency_features])
#     dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(concatenated_features)
#     dense1 = Dropout(0.5)(dense1)
#     output = Dense(6, activation='softmax')(dense1)

#     model = Model(inputs=inputs, outputs=output)
#     return model

# # Create the model
# model_with_saliency = build_model()

# # Attempt to load weights
# weights_path = 'CNN_skin_classifier_weights.weights.h5'
# try:
#     model_with_saliency.load_weights(weights_path)
#     print("Weights loaded successfully.")
# except ValueError as e:
#     print(f"Error loading weights: {e}")

# # Print the model summary to verify
# model_with_saliency.summary()

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
import matplotlib.pyplot as plt
from io import StringIO
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB4, VGG16, ResNet50, InceptionResNetV2
import tensorflow as tf
import keras
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2

# Define the model (this should match your previous model definition)
model_skin_cnn = build_model()
try:
    model_skin_cnn.load_weights('CNN_skin_classifier_weights.weights.h5')
except ValueError as e:
    st.error(f"Error loading weights for CNN_skin_cnn: {e}")
    
    
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

# Preprocess function for saliency and image
def preprocess_image(img, model_name):
    input_size = (224, 224)
    img = image.load_img(img, target_size=input_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Add dummy saliency map if needed
    saliency_map = np.zeros_like(img)
    combined_input = np.concatenate((img, saliency_map), axis=-1)
    return combined_input



        
        
        
        
        
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

def melanoma_detection():
    st.title('Melanoma Detection')

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
        st.write("Classifying...")

        try:
            combined_input = preprocess_image(uploaded_file, selected_model)

            # Preprocess based on model
            if selected_model in preprocess_dict:
                combined_input = preprocess_dict[selected_model](combined_input)

            # Model prediction
            model = model_dict[selected_model]
            prediction = model.predict(combined_input)

            if prediction.shape[-1] > 1:
                predicted_class = np.argmax(prediction, axis=-1)
                confidence = np.max(prediction)
            else:
                threshold = 0.5
                predicted_class = (prediction > threshold).astype(int)
                confidence = prediction[0][0]

            if model_type == 'Skin Image Models':
                result = 'Melanoma' if predicted_class[0] == 1 else 'Non-Melanoma'
            else:
                result = 'Malignant' if predicted_class[0] == 1 else 'Benign'

            st.write(f"Prediction: {result}")
            st.write(f"Confidence: {confidence:.2f}")

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

    
# # Define class labels for multi-class classification
# skin_labels = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']

# @st.cache_resource
# def load_models():
#     models = {
#         'skin': {},
#         'derm': {}
#     }

#     # Initialize model variables
#     model_skin_cnn = None
#     model_skin_vgg16 = None
#     model_skin_resnet50 = None
#     model_skin_efficientnet = None
#     model_skin_inceptionresnetv2 = None
    
#     model_derm_cnn = None
#     model_derm_vgg16 = None
#     model_derm_resnet50 = None
#     model_derm_efficientnet = None
#     model_derm_inceptionresnetv2 = None

#     # Google Drive file IDs
#     files = {
#         'DM_melanoma_cnn_with_saliency.keras': '14K_mzVdlE0389-z1O2PRoVEFX6KZ7nWs',
#         'DM_vgg16_model_with_saliency.keras': '1wCXdV3nhNcxcNWr0WKc7cX8kKM4vaTzt',
#         'DM_best_ResNet50_model.keras': '11rVWX0nj193MkaRA_51kqh8stmN-axj_',
#         'DM_efficientnetb4_model_with_saliency.keras': '1WByq-kIVyzfDXOdI3nH2Y6saSaHu8Evm',
#         'DM_InceptionResNetV2_model.keras': '1z60vHbKeegg0Frfb-QNhZpI-pj3KmbiD',
#         'CNN_skin_classifier_weights.weights.h5': '17jF5po5RQwrzG10Yyxj6TJBn_-hoVakE',
#         'best_VGG16_weights.weights.h5': '1iJz12SpdkSi_TVrz4rgNB4G16xhRGJgi',
#         'best_ResNet50_weights.weights.h5': '1-4MdLCmA6l30Of_ZuCO_SNpLveTClcsX',
#         'best_EfficientNetB4_weights.weights.h5': '1-0ytyTEkcPLYaOf4AGJ1VPQ5EZjPJudr',
#         'best_InceptionResNetV2_weights.weights.h5': '1WAZBPiYVCHp6Lgu_1d9bOTzterk5o5vj',
#     }

#     # Function to download and load model
#     def download_and_load_model(file_name, build_func=None, input_shape=None, num_classes=None):
#         url = f"https://drive.google.com/uc?id={files[file_name]}&export=download"
#         output = file_name
#         gdown.download(url, output, quiet=False)

#         if file_name.endswith('.keras'):
#             return load_model(output)
#         elif file_name.endswith('.weights.h5'):
#             if build_func:
#                 model = build_func(input_shape=input_shape, num_classes=num_classes)
#                 model.load_weights(output)
#                 return model
#             else:
#                 st.error(f"Build function not provided for {file_name}")
#                 return None

#     # Load dermoscopy image models
#     try:
#         model_derm_cnn = download_and_load_model('DM_melanoma_cnn_with_saliency.keras')
#         model_derm_vgg16 = download_and_load_model('DM_vgg16_model_with_saliency.keras')
#         model_derm_resnet50 = download_and_load_model('DM_best_ResNet50_model.keras')
#         model_derm_efficientnet = download_and_load_model('DM_efficientnetb4_model_with_saliency.keras')
#         model_derm_inceptionresnetv2 = download_and_load_model('DM_InceptionResNetV2_model.keras')
#     except Exception as e:
#         st.error(f"Error loading dermoscopy image models: {e}")

#     # Load skin image models
#     failed_models = []

#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     # CNN model for skin images
#     try:
#         logger.info("Attempting to download CNN model for skin images...")
#         url = "https://drive.google.com/uc?id=17jF5po5RQwrzG10Yyxj6TJBn_-hoVakE&export=download"
#         output = "CNN_skin_classifier_weights.weights.h5"
#         gdown.download(url, output, quiet=False)
        
#         logger.info("Building CNN model...")
#         model_skin_cnn = build_model(input_shape=(224, 224, 6))
        
#         logger.info("Loading weights for CNN model...")
#         model_skin_cnn.load_weights(output)
        
#         models['skin']['CNN'] = model_skin_cnn
#         logger.info("CNN model for skin images loaded successfully.")
#     except Exception as e:
#         logger.error(f"Failed to load CNN model for skin images: {str(e)}")
#         failed_models.append("skin_CNN")



#     try:
#         model_skin_vgg16 = download_and_load_model('best_VGG16_weights.weights.h5', build_model_with_saliency_Vgg, input_shape=(224, 224, 6), num_classes=6)
#     except Exception as e:
#         st.error(f"Error loading VGG16 skin model: {e}")

#     try:
#         model_skin_resnet50 = download_and_load_model('best_ResNet50_weights.weights.h5', build_model_with_saliency_Res, input_shape=(224, 224, 6), num_classes=6)
#     except Exception as e:
#         st.error(f"Error loading ResNet50 skin model: {e}")

#     try:
#         model_skin_efficientnet = download_and_load_model('best_EfficientNetB4_weights.weights.h5', build_model_with_saliency_Eff, input_shape=(380, 380, 6), num_classes=6)
#     except Exception as e:
#         st.error(f"Error loading EfficientNetB4 skin model: {e}")

#     try:
#         model_skin_inceptionresnetv2 = download_and_load_model('best_InceptionResNetV2_weights.weights.h5', build_model_with_saliency_Inc, input_shape=(299, 299, 6), num_classes=6)
#     except Exception as e:
#         st.error(f"Error loading InceptionResNetV2 skin model: {e}")

#     return {
#         'derm': {
#             'CNN': model_derm_cnn,
#             'VGG16': model_derm_vgg16,
#             'ResNet50': model_derm_resnet50,
#             'EfficientNetB4': model_derm_efficientnet,
#             'InceptionResNetV2': model_derm_inceptionresnetv2
#         },
#         'skin': {
#             'CNN': model_skin_cnn,
#             'VGG16': model_skin_vgg16,
#             'ResNet50': model_skin_resnet50,
#             'EfficientNetB4': model_skin_efficientnet,
#             'InceptionResNetV2': model_skin_inceptionresnetv2
#         }
#     }, model_skin_cnn, model_skin_vgg16, model_skin_resnet50, model_skin_efficientnet, model_skin_inceptionresnetv2