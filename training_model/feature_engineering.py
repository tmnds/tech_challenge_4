import yfinance as yf
import pandas as pd

def create_df_for_multi_companies(
        raw_data_multiindex:pd.MultiIndex, 
        stock_var:str, 
        companies:list[str]):
    
    raw_values = {}
    for company in companies:
        # Transform MultiIndex DF into SingleIndex DF for each company with all prices (Open, Close... Adj. Close)
        raw_data_by_company = raw_data_multiindex.xs(key=company, level='Ticker', axis=1, drop_level=False)
        # Grab only the Datetime and the desired price (e.g., colums -> [Datetime | Open])
        raw_data_series = raw_data_by_company[(stock_var,company)]
        # Make a dictionary of type {company[x] : [Open Values 0, 1, ... N] ... }
        raw_values[company] = raw_data_series.values
    # Create 'df' Dataframe only with "Datetime" Column
    df = pd.DataFrame({'Datetime':raw_data_series.index})
    # Append all columns with values for each company, resulting in [Datetime | Company_values[x] ...]
    df = df.assign(**raw_values)
    return df

def get_finance_df(
        companies:str|list[str], 
        start_date:str, 
        end_date:str, 
        stock_var:str='Adj Close') -> pd.DataFrame:
    
    try:
        # In case companies is a string, transform into a list of 1 object, like: ['company']
        if not isinstance(companies,list):
            companies:list[str] = [companies]
        
        # Download data using yfinance
        raw_data_multiindex:pd.MultiIndex = yf.download(tickers=companies, start=start_date, end=end_date)

        # Transform the raw data into a suitable DF
        df = create_df_for_multi_companies(raw_data_multiindex, stock_var, companies)

        return df.interpolate(method='linear')
    
    except:
            Exception("Unexpected error: something wrong occurred while creating the finance DF")



def split_train_test_valid_df(df, horizon_pred=1, seq_length=30, size_train_percent=0.75, size_test_percent=0.2, create_valid_df = False):
    if create_valid_df == True:    
        N = len(df)-seq_length-horizon_pred
        Ntrain = int(size_train_percent*N)
        Ntest = int(size_test_percent*N)
        
        df_train = df.loc[0:Ntrain]
        df_test = df.loc[Ntrain+1:Ntrain+Ntest]
        df_valid = df.loc[Ntrain+Ntest+1:]

        Nvalid = len(df_valid)
        if (Nvalid < (seq_length+horizon_pred+1)):
            raise ValueError(f"""At least (seq_length+horizon_pred+1) samples are needed for validation. You have seq_length={seq_length}, horizon_pred={horizon_pred} and Nvalid={Nvalid} samples, with Ntrain={Ntrain}, Ntest={Ntest} and N={N}. Try changing the prediction horizon, the sequence/window length, or increasing the number of samples""")

        return (df_train, df_test, df_valid)
    else:
        N = len(df)
        Ntrain = int(size_train_percent*N)

        df_train = df.loc[0:Ntrain]
        df_test = df.loc[Ntrain:]

        return (df_train, df_test)

def shift_drop_na_in_xy(df, company_inputs, company_output, horizon_pred):
    new_df = pd.concat([df[company_inputs], df[company_output].rename("target").shift(-horizon_pred)],axis='columns').dropna()
    X = new_df[company_inputs].values
    y = new_df["target"].values
    return X, y