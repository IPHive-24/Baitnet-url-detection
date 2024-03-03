from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load your deep learning model
model = load_model(r"C:\Users\manoj\OneDrive\Desktop\FINAL\1.h5")

# Load the tokenizer and other required objects
tokenizer = Tokenizer(num_words=20000, char_level=True)
 # Replace with your actual word_index

@app.route('/')
def index():
    return render_template(r"index.html")

@app.route('/check_url', methods=['POST'])
def check_url():
    if request.method == 'POST':
        url = request.form['url']

        # Tokenize and pad the input URL
        sequences = tokenizer.texts_to_sequences([url])
        X_padded = pad_sequences(sequences, maxlen=128, padding='post', truncating='post', value=)
        X_padded = np.array(X_padded)

        # Make a prediction using your model
        prediction = model.predict(X_padded)

        # Define a threshold (e.g., 0.5) for classifying as phishing or not
        threshold = 0.5
        result = "Phishing" if prediction >= threshold else "Not Phishing"

        return jsonify({'result': result})
    
if __name__ == '__main__':
    app.run(debug=True)
