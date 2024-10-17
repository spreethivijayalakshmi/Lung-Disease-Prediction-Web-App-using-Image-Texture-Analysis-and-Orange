import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
import joblib
from flask import Flask, request, render_template
import os
import Orange
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load models
pca_model = joblib.load('new_trained_pca_model.pkl')
orange_model = joblib.load('neaural_model.pkcls')


def preprocess_image(image_path):
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image from {image_path}")

    equalized_image = cv2.equalizeHist(image)
    resized_img = cv2.resize(equalized_image, (128, 128))

    # Extract HOG features
    features, _ = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    print(f"Extracted {len(features)} HOG features")  # Debugging

    # Ensure the features are the correct size (8100)
    if len(features) != 8100:
        raise ValueError(f"Expected 8100 features, but got {len(features)}.")

    return features


def preprocess_with_pca(image_path, pca_model):
    # Preprocess the image to extract HOG features
    features = preprocess_image(image_path)

    # Apply PCA transformation to reduce features
    reduced_features = pca_model.transform([features])
    print(f"PCA reduced features shape: {reduced_features.shape}")  # Debugging

    # Create a DataFrame for the reduced features
    pca_feature_names = [f'PCA_Feature_{i}' for i in range(reduced_features.shape[1])]
    df_reduced = pd.DataFrame(reduced_features, columns=pca_feature_names)

    return df_reduced


# Mapping prediction numbers to labels
label_mapping = {
    0.0: 'Normal',
    1.0: 'Lung Opacity',
    2.0: 'Viral Pneumonia',
}

# Recommendations for each condition
recommendations = {
    'Normal': 'Congratulations You have Healthy lungs,To maintain healthy lungs, it’s essential to avoid smoking and secondhand smoke, as they can significantly harm lung function. Engaging in regular exercise, such as walking or cycling, can enhance lung capacity and efficiency. A balanced diet rich in fruits and vegetables, particularly those high in antioxidants, supports overall lung health. Staying well-hydrated is also crucial, as it helps keep mucus thin, facilitating optimal lung performance. Additionally, practicing deep breathing exercises can strengthen lung function, while minimizing exposure to air pollutants and allergens helps maintain clear airways and promotes better respiratory health.',
    'Lung Opacity': 'You have diagnosed with lung opacity,This is not a final report,It is important to consult a healthcare professional for further evaluation and diagnosis. Lung opacity can indicate various conditions, such as pneumonia, fluid accumulation, or other lung diseases. A detailed medical history, imaging tests, and possibly a biopsy may be required to determine the underlying cause. Avoid smoking or exposure to air pollutants, and follow any prescribed medications or treatments. Early diagnosis and treatment are crucial to managing the condition effectively.',
    'Viral Pneumonia': 'You have diagnosed with viral pneumonia,This is not a final report,For viral pneumonia, it is essential to seek medical attention immediately for proper diagnosis and treatment. Antiviral medications may be prescribed depending on the virus causing the infection. Rest and hydration are critical for recovery, along with over-the-counter medications to relieve symptoms like fever and cough. Patients should avoid smoking and exposure to irritants to reduce lung stress. It is also important to maintain good hygiene, such as frequent hand washing and wearing masks in public, to prevent the spread of infections. In severe cases, hospitalization may be required for oxygen therapy and supportive care.',
}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            try:
                # Ensure the uploads folder exists
                upload_folder = os.path.join(os.getcwd(), 'uploads')
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)

                # Use secure_filename to sanitize the filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)

                # Print for debugging
                print(f"Saving file to: {file_path}")

                # Save the file
                file.save(file_path)
            except Exception as e:
                return f"Error saving file: {str(e)}"

            # Preprocess the image with PCA
            try:
                print(f"Processing image at path: {file_path}")  # Debugging

                reduced_features = preprocess_with_pca(file_path, pca_model)
                print(f"Reduced features DataFrame: {reduced_features}")  # Debugging
            except Exception as e:
                return f"Error processing image: {str(e)}"

            # Prepare data for the Orange model
            try:
                orange_data = Orange.data.Table.from_list(
                    domain=orange_model.domain,
                    rows=reduced_features.values.tolist()
                )
                print(f"Orange data: {orange_data}")  # Debugging
            except Exception as e:
                return f"Error preparing Orange data: {str(e)}"

            # Make a prediction using the Orange model
            try:
                prediction = orange_model(orange_data)
                print(f"Prediction object: {prediction}")  # Debugging

                # Extract the predicted class
                predicted_class = float(prediction[0])
                print(f"Predicted class: {predicted_class}")  # Debugging
            except Exception as e:
                return f"Error predicting: {str(e)}"

            # Map the prediction to the label and recommendation
            predicted_label = label_mapping.get(predicted_class, "Unknown")
            recommendation_text = recommendations.get(predicted_label, "No recommendations available.")

            return render_template('result.html', prediction=predicted_label, recommendation=recommendation_text)

    return render_template('home.html')




if __name__ == '__main__':
    app.run(debug=True)
