from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import keras.saving as sg # type: ignore
import os
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model (this could be done once on app startup)
loaded_model = sg.load_model('final.keras')

# Directory to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check the allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page where the user uploads the image
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        # If no file is selected, redirect to the home page
        if file.filename == '':
            return redirect(request.url)
        
        # If the file is allowed, process the image and make prediction
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Process the image and make a prediction
            img = Image.open(filepath).resize((28, 28))
            img_array = np.array(img).reshape(-1, 28, 28, 3)  # Reshaping for model input

            # Predict the classes
            r = loaded_model.predict(img_array)
            r = r.tolist()

            # Class probabilities and descriptions for diabetic retinopathy
            class_probabilities = r[0]
            predicted_class_index = np.argmax(class_probabilities)  # Index of the highest probability
            predicted_probability = class_probabilities[predicted_class_index]

            class_descriptions = {
                0: "No DR (No diabetic retinopathy detected)",
                1: "Mild DR (Early signs of diabetic retinopathy, minimal damage to the retina)",
                2: "Moderate DR (More severe damage, but vision loss is not imminent)",
                3: "Severe DR (Advanced stage with significant damage to the retina and risk of vision loss)",
                4: "Proliferative DR (Most severe stage with abnormal blood vessels growing in the retina, leading to potential vision loss)"
            }

            # Get the predicted class description
            predicted_class_description = class_descriptions[predicted_class_index]

            # Pass the predicted class and description to the result page
            return render_template('result.html', 
                                   predicted_class_index=predicted_class_index, 
                                   predicted_class_description=predicted_class_description, 
                                   predicted_probability=predicted_probability)

    return render_template('index.html')


# Route to show the result after the prediction
@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
