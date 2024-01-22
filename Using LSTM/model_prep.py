import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv('labeled_data.csv')
df.head()

df.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], inplace=True)
df["class"] = df['class'].map({0: 'Hate Speech', 1: 'Offensive Speech', 2: 'No Hate and Offensive Speech'})
df.head()

# Clean the text
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

def clean(text):
    text = str(text).lower()
    text = re.sub('[,?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<,?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)

# Encode labels
label_encoder = LabelEncoder()
df["class_encoded"] = label_encoder.fit_transform(df["class"])

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["tweet"])
x = tokenizer.texts_to_sequences(df["tweet"])
x = pad_sequences(x)

print("Maximum sequence length:", x.shape[1])

# Map classes to binary labels
df["binary_class"] = df["class"].map({'Hate Speech': 1, 'Offensive Speech': 1, 'No Hate and Offensive Speech': 0})

y = np.array(df["binary_class"])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=x.shape[1]))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import pickle
def save_model(model, model_path='hate_speech_model'):
    model.save(model_path)

def save_tokenizer(tokenizer, file_path='tokenizer.pickle'):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_label_encoder(lbl_encoder, file_path='label_encoder.pickle'):
    with open(file_path, 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

save_model(model)
save_tokenizer(tokenizer)
save_label_encoder(label_encoder)