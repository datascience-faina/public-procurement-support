# Model building & evaluation



# One-Hot Encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



# Categorical columns encoding

def encode(df):
    """
    One-Hot Encoding for train and test.
    
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # numeric
    df_num = df[numeric_cols].reset_index(drop=True)

    # transform train/test
    df_ohe = encoder.transform(df[categorical_cols])

    ohe_columns = encoder.get_feature_names(categorical_cols)

    # final dataframe
    df_cat = pd.DataFrame(df_ohe, columns=ohe_columns)
    df_encoded = pd.concat([df_num, df_cat], axis=1)
   
    return df_encoded