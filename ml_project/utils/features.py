def prepare_data(df):
    df = df.rename(columns={"TABLE": "TABLE_PCT"})
    median_price = df["PRICE"].median()
    df["PRICE_CATEGORY"] = (df["PRICE"] > median_price).astype(int)
    return df
