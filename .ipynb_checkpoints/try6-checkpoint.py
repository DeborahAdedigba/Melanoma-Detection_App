import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import yaml
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load configuration from YAML
with open('melanoma_detection_app.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set page configuration
st.set_page_config(page_title=config['app']['title'], page_icon=config['app']['icon'], layout=config['app']['layout'])



# CNN
def build_model(input_shape=(224, 224, 6)):  # 6 channels for image + saliency
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    def create_cnn(input_tensor):
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(0.5)(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv5)
        conv5 = BatchNormalization()(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        pool5 = Dropout(0.5)(pool5)
        flatten = Flatten()(pool5)
        return flatten

    image_features = create_cnn(image_input)
    saliency_features = create_cnn(saliency_input)
    concatenated_features = tf.keras.layers.Concatenate()([image_features, saliency_features])
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(concatenated_features)
    dense1 = Dropout(0.5)(dense1)
    output = Dense(6, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=output)
    return model

# EfficientB4
def build_model_with_saliency_Eff(input_shape=(380, 380, 6), num_classes=6):
    base_model = EfficientNetB4(include_top=False, weights=None, input_shape=(380, 380, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# VGG16


def build_model_with_saliency_Vgg(input_shape=(224, 224, 6), weights_path=None, num_classes=6):
    base_model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    if weights_path is not None:
        base_model.load_weights(weights_path, by_name=True)
    base_model.trainable = False  # Freeze the base model

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

#Resnet50
def build_model_with_saliency_Res(input_shape=(224, 224, 6), num_classes=6):
    base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# InceptionResNetV2


def build_model_with_saliency_Inc(input_shape=(299, 299, 6), num_classes=6):
    base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=(299, 299, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model


# Load models
@st.cache_resource
def load_models():
    models = {
        'skin': {},
        'derm': {}
    }
    
    for category in ['skin', 'derm']:
        for model_name, model_info in config['models'][category].items():
            try:
                if category == 'skin':
                    if model_name == 'CNN':
                        model = build_model()  # Use the CNN model function
                    elif model_name == 'EfficientNetB4':
                        model = build_model_with_saliency_Eff()  # Use the EfficientNetB4 model function
                    elif model_name == 'InceptionResNetV2':
                        model = build_model_with_saliency_Inc()  # Use the InceptionResNetV2 model function
                    elif model_name == 'ResNet50':
                        model = build_model_with_saliency_Res()  # Use the ResNet50 model function
                    elif model_name == 'VGG16':
                        model = build_model_with_saliency_Vgg()  # Use the VGG16 model function
                    
                    model.load_weights(model_info['path'])  # Load the weights
                    models[category][model_name] = model

                else:
                    models[category][model_name] = load_model(model_info['path'])  # Load full model for 'derm'

            except Exception as e:
                st.error(f"Error loading {category} {model_name} model: {e}")
    
    return models


# Melanoma detection function
def melanoma_detection():
    st.title('Melanoma Detection')
    
    # Model selection using tabs
    tab1, tab2 = st.tabs(["Skin Image Models", "Dermoscopy Image Models"])
    
    loaded_models = load_models()
    
    with tab1:
        st.header("Skin Image Models")
        models = list(config['models']['skin'].keys())
        model_dict = loaded_models['skin']
    
    with tab2:
        st.header("Dermoscopy Image Models")
        models = list(config['models']['derm'].keys())
        model_dict = loaded_models['derm']
    
    selected_model = st.selectbox('Select Model', models)
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        try:
            # Process uploaded image
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            
            # Preprocess based on model
            preprocess_dict = {
                'VGG16': preprocess_vgg16,
                'ResNet50': preprocess_resnet50,
                'EfficientNetB4': preprocess_efficientnet,
                'InceptionResNetV2': preprocess_inceptionresnetv2
            }
            if selected_model in preprocess_dict:
                img = preprocess_dict[selected_model](img)
            
            # Model prediction
            model = model_dict.get(selected_model)
            if model is None:
                st.error("Selected model is not available.")
            else:
                prediction = model.predict(img)
                
                # Handle the prediction output
                if tab1:  # Multi-class output for skin images
                    predicted_class = np.argmax(prediction, axis=-1)
                    confidence = np.max(prediction)
                    result = config['skin_labels'][predicted_class[0]]
                else:  # Binary output for dermoscopy images
                    threshold = 0.5
                    predicted_class = (prediction[:, 0] > threshold).astype(int)
                    confidence = prediction[0, 0]
                    result = 'Malignant' if predicted_class[0] == 1 else 'Benign'
                
                st.write(f"Prediction: {result}")
                st.write(f"Confidence: {confidence:.2f}")
                
                st.write("\nPlease note:")
                st.write(config['disclaimer_text'])
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please upload an image.")

# Model performance page
def model_performance_page():
    st.title("Model Performance")
    st.markdown("This section allows you to explore the performance of various models used for melanoma detection.")
    
    model_type = st.sidebar.selectbox('Select Model Type', ['Skin Image Models', 'Dermoscopy Image Models'])
    metrics = config['metrics']['skin'] if model_type == 'Skin Image Models' else config['metrics']['derm']
    model_name = st.sidebar.selectbox('Select Model', list(metrics.keys()))
    
    # Add debugging code here
    models = load_models()  # Load the models
    st.write("Loaded models for 'skin':", models['skin'].keys())  # Debug: Show loaded models
    st.write("Attempting to load model:", model_name)  # Debug: Show the model being accessed
    
    # Access the selected model
    model = models['skin' if model_type == 'Skin Image Models' else 'derm'][model_name]

    # Continue with the rest of the function as is
    tab1, tab2, tab3, tab4 = st.tabs(["Model Summary", "Performance Metrics", "Confusion Matrix", "ROC Curve"])
    
    with tab1:
        st.header(f"{model_type} - {model_name} Model Summary")
        summary_string = StringIO()
        model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
        st.text(summary_string.getvalue())
    
    with tab2:
        st.header("Performance Metrics")
        for metric, value in metrics[model_name].items():
            st.write(f"**{metric}**: {value}")
    
    with tab3:
        st.header("Confusion Matrix")
        matrix = config['confusion_matrices']['skin' if model_type == 'Skin Image Models' else 'derm'][model_name]
        df_cm = pd.DataFrame(matrix)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name} ({model_type})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(plt)
        
    with tab4:
        st.header(f"ROC Curve for {model_name}")
        num_classes = 6 if model_type == 'Skin Image Models' else 2
        fig = plot_roc_curve(model, model_name, num_classes)
        st.plotly_chart(fig)

# Visualization page
import plotly.express as px
import pandas as pd

def visualize_data():
    st.title('Visualizations')
    st.markdown("This section provides interactive visualizations to help you understand the distribution of melanoma cases, age, and gender in the dataset.")

    # Sidebar selection
    visualization_type = st.sidebar.selectbox('Select Visualization', ['Skin', 'Dermoscopy'])

    if visualization_type == 'Skin':
        # Skin Visualizations
        st.subheader("Skin Image Dataset Visualizations")

        # Create a sample dataframe based on the confusion matrix
        skin_matrix = config['confusion_matrices']['skin']['ResNet50']  # Using ResNet50 as an example
        skin_labels = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        skin_data = []
        for i, row in enumerate(skin_matrix):
            for j, value in enumerate(row):
                skin_data.append({'Actual': skin_labels[i], 'Predicted': skin_labels[j], 'Count': value})
        df_skin = pd.DataFrame(skin_data)

        # Heatmap of confusion matrix
        fig_heatmap = px.imshow(skin_matrix, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=skin_labels, y=skin_labels,
                                title="Confusion Matrix Heatmap for Skin Images (ResNet50)")
        st.plotly_chart(fig_heatmap)

        # Bar chart of class distribution
        class_distribution = df_skin.groupby('Actual')['Count'].sum().reset_index()
        fig_bar = px.bar(class_distribution, x='Actual', y='Count', 
                         title="Distribution of Skin Lesion Classes")
        st.plotly_chart(fig_bar)

    else:
        # Dermoscopy Visualizations
        st.subheader("Dermoscopy Image Dataset Visualizations")

        # Create a sample dataframe based on the confusion matrix
        derm_matrix = config['confusion_matrices']['derm']['EfficientNetB4']  # Using EfficientNetB4 as an example
        derm_labels = ['Benign', 'Malignant']
        derm_data = []
        for i, row in enumerate(derm_matrix):
            for j, value in enumerate(row):
                derm_data.append({'Actual': derm_labels[i], 'Predicted': derm_labels[j], 'Count': value})
        df_derm = pd.DataFrame(derm_data)

        # Heatmap of confusion matrix
        fig_heatmap = px.imshow(derm_matrix, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=derm_labels, y=derm_labels,
                                title="Confusion Matrix Heatmap for Dermoscopy Images (EfficientNetB4)")
        st.plotly_chart(fig_heatmap)

        # Pie chart of class distribution
        class_distribution = df_derm.groupby('Actual')['Count'].sum().reset_index()
        fig_pie = px.pie(class_distribution, values='Count', names='Actual', 
                         title="Distribution of Dermoscopy Image Classes")
        st.plotly_chart(fig_pie)

    # Add a note about the data source
    st.info("Note: These visualizations are based on the confusion matrices provided in the configuration. They represent model performance rather than the raw dataset distribution.")

# Educational resources page
def educational_resources():
    st.title('Educational Resources')
    st.markdown("A curated list of resources to deepen your understanding of melanoma and the technologies used for detection.")
    
    for resource in config['educational_resources']:
        st.markdown(f"- [{resource['title']}]({resource['url']})")

# FAQ page
def faq_section():
    st.title('FAQs')
    
    for faq in config['faqs']:
        st.subheader(faq['question'])
        st.write(faq['answer'])



# Function to send feedback via email
def send_email(name, email, message):
    sender_email = "debbydawn16@gmail.com"  # Replace with your Gmail address
    sender_password = "fcgd zzhr szgf izia"  # Replace with your app password

    recipient_email = "debbydawn16@gmail.com"
    
    subject = "New Feedback from Melanoma Detection App"
    
    # Create the email body
    body = f"""
    You have received a new feedback message:

    Name: {name}
    Email: {email}
    Message: {message}
    """

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish a secure session with Gmail's outgoing SMTP server using your Gmail account
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Use TLS to add security
        server.login(sender_email, sender_password)
        
        # Send email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.close()

        st.success("Your message has been sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Feedback form
def feedback_form():
    st.title('Feedback and Contact')
    st.write("If you have any feedback, questions, or need support, please don't hesitate to reach out to us.")

    # Form input fields
    with st.form(key='feedback_form'):
        name = st.text_input('Name')
        email = st.text_input('Email')
        message = st.text_area('Message')

        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # Process the feedback form
    if submit_button:
        if not name or not email or not message:
            st.error("Please fill out all fields.")
        else:
            send_email(name, email, message)

# Main app
def main():
    st.sidebar.title('Navigation')
    
    pages = {
        "Introduction": lambda: st.markdown("Welcome to the Melanoma Detection App."),
        "Model Performance": model_performance_page,
        "Visualizations": visualize_data,
        "Melanoma Detection": melanoma_detection,
        "Educational Resources": educational_resources,
        "FAQs": faq_section,
        "Feedback and Contact": feedback_form
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()
    
    st.sidebar.markdown("Disclaimer: This tool is for educational purposes only.")

if __name__ == '__main__':
    main()
