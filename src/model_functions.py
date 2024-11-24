import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def make_predictions(X, y, seq_length, batch_size, scaler, model):

    generator = TimeseriesGenerator(X, y, length=seq_length, batch_size=batch_size)

    scaled_X = []
    y_pred = []
    for k in range(len(generator)):
        X_temp, _ = generator[k]
        scaled_X = scaler.transform(X_temp.reshape(-1,1)).reshape(len(X_temp),seq_length)

        y_pred_scaled = model.predict(scaled_X)
        y_pred.append(scaler.inverse_transform(y_pred_scaled.reshape(-1,1)))
        
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred.reshape(1,-1)