import pickle
def save_model(model, model_path='hate_speech_model.h5'):
    model.save(model_path)

def save_tokenizer(tokenizer, file_path='tokenizer.pickle'):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_label_encoder(lbl_encoder, file_path='label_encoder.pickle'):
    with open(file_path, 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)