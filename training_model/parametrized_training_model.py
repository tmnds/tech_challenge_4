from feature_engineering import get_finance_df
from feature_engineering import split_train_test_valid_df, shift_drop_na_in_xy

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.regularizers import l1, l2, l1_l2

stock_var = 'Adj Close'

def custom_lstm_model(seq_length, n_inputs):
    model = Sequential([
            Input((seq_length, n_inputs)),
            LSTM(units=100, activation='relu',
            ),
            Dropout(rate=0.1),
            Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mse')

    return model
    

def parametrized_training(company_inputs, company_output, start_date, end_date, train_ratio,
                          horizon_pred, seq_length, batch_size, n_epochs):

    df = get_finance_df(company_inputs, start_date, end_date, stock_var)
    df = df.interpolate(method='linear')

    n_inputs = len(company_inputs)    

    df_train, df_test = split_train_test_valid_df(df=df, 
        horizon_pred=horizon_pred, seq_length=seq_length, 
        size_train_percent=train_ratio, create_valid_df=False
    )

    X_train, y_train = shift_drop_na_in_xy(df_train, company_inputs, company_output, horizon_pred=horizon_pred)
    X_test, y_test = shift_drop_na_in_xy(df_test, company_inputs, company_output, horizon_pred=horizon_pred)

    # Normalize the Price column
    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()

    scaled_X_train = scalerX.fit_transform(X_train)
    scaled_y_train = scalery.fit_transform(y_train.reshape(-1, 1))

    scaled_X_test = scalerX.transform(X_test)
    scaled_y_test = scalery.transform(y_test.reshape(-1, 1))

    # Initialize generator with multivariable input and single target
    generator_train = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=seq_length, batch_size=batch_size)
    generator_test = TimeseriesGenerator(scaled_X_test, scaled_y_test, length=seq_length, batch_size=batch_size)

    model = custom_lstm_model(seq_length, n_inputs)

    return (scalerX, scalery, generator_train, generator_test, model)