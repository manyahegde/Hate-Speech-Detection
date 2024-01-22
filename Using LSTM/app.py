from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

app = Flask(__name__)

# Load the saved model, tokenizer, and label encoder
def load_saved_objects():
    model = load_model('hate_speech_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as ecn_file:
        label_encoder = pickle.load(ecn_file)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_saved_objects()

def clean_input(text):
    # Your cleaning logic here
    return text

def predict_hate_speech(input_text):
    cleaned_text = clean_input(input_text)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    padded_text = pad_sequences(tokenized_text, maxlen=25)  # Specify the max length used during training
    prediction_prob = model.predict(padded_text)[0, 0]

    # Adjust the threshold
    threshold = 0.5
    predicted_class = "Hate Speech" if prediction_prob > threshold else "Not a Hate Speech"

    return predicted_class, prediction_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        predicted_class, prediction_prob = predict_hate_speech(input_text)
        return render_template('index.html', prediction=predicted_class, prob=prediction_prob)

if __name__ == '__main__':
    app.run(debug=True)
