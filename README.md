# Melanoma Detection App

## Overview

This Melanoma Detection App is an interactive web application designed to assist in the detection and understanding of melanoma, a serious form of skin cancer. The app utilizes advanced deep learning models to analyze skin and dermoscopy images, providing educational resources and visualizations to enhance understanding of melanoma.

## Features

- **Multiple Model Support**: Includes various deep learning models (CNN, VGG16, ResNet50, EfficientNetB4, InceptionResNetV2) for both skin and dermoscopy image analysis.
- **Interactive Image Upload**: Users can upload their own skin or dermoscopy images for analysis.
- **Model Performance Metrics**: Displays detailed performance metrics including accuracy, precision, recall, and AUC for each model.
- **Visualizations**: Offers interactive data visualizations to understand the distribution of melanoma cases, age, and gender in the dataset.
- **Educational Resources**: Provides curated links to academic articles, websites, and research papers on melanoma and related technologies.
- **FAQs Section**: Answers common questions about the app and melanoma detection.
- **Feedback System**: Includes a form for users to submit feedback or questions.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Deep Learning**: TensorFlow, Keras
- **Data Visualization**: Plotly, Seaborn
- **Image Processing**: Pillow

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/DeborahAdedigba/Melanoma-Detection.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

Navigate through the app using the sidebar menu:

1. **Introduction**: Learn about the app and melanoma.
2. **Model Performance**: Explore the performance metrics of different models.
3. **Visualizations**: View interactive charts and graphs of melanoma data.
4. **Melanoma Detection**: Upload images for analysis.
5. **Educational Resources**: Access curated learning materials.
6. **FAQs**: Find answers to common questions.
7. **Feedback and Contact**: Submit your feedback or questions.

## Data Sources

- PH2 Dataset for skin images
- ISIC 2016 Dataset for dermoscopy images

## Disclaimer

This app is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Contributions to improve the app are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

  GNU GENERAL PUBLIC LICENSE 
  Version 3, 29 June 2007

## Contact

For any queries or support, please use the Feedback and Contact form within the app or reach out to 2adedd38@solent.ac.uk.



