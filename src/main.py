import os
import sys
import pandas as pd

import src.data_handling as data_handling
from src._utils import main_logger


def run_lakehouse(ticker:str = 'NVDA', should_local_save: bool = True):
    df = None

    # extract
    stock_price_data = data_handling.extract_daily_stock_data(ticker=ticker)

    # bronze
    bronze_s3_path = data_handling.bronze.load_to_s3(data=stock_price_data, ticker=ticker)

    # start the spark session
    spark = data_handling.config_and_start_spark_session()

    # silver
    bronze_delta_table = spark.read.json(bronze_s3_path, multiLine=True)
    silver_df = data_handling.silver.process(delta_table=bronze_delta_table, spark=spark)
    silver_s3_path = data_handling.silver.load(df=silver_df, ticker=ticker)

    # gold
    silver_delta_table = data_handling.retrieve_delta_table(spark=spark, s3_path=silver_s3_path)
    gold_df = data_handling.gold.process(delta_table=silver_delta_table, spark=spark)
    gold_s3_path = data_handling.gold.load(df=gold_df, ticker=ticker)

    if gold_df and should_local_save:
        # save gold_df as pandas df in parquet / csv file
        df = gold_df.toPandas()
        df['dt'] = pd.to_datetime(df['dt'])

        df_folder_path = os.path.join('data', 'df', TICKER)
        os.makedirs(df_folder_path, exist_ok=True)

        df_parquet_file_name = 'df.parquet'
        df_csv_file_name = 'df.csv'

        df.to_parquet(os.path.join(df_folder_path, df_parquet_file_name), index=False)
        df.to_csv(os.path.join(df_folder_path, df_csv_file_name), index=False)

        main_logger.info(f'... pyspark df successfully convert to pandas df and saved to {os.path.join(df_folder_path, df_parquet_file_name)} and {os.path.join(df_folder_path, df_csv_file_name)}. returning pandas df for donwstream use ...')

    if not gold_s3_path: main_logger.error('... failed to load gold df. return none ...')

    return df, spark



if __name__ == '__main__':
    TICKER = sys.argv[1] if len(sys.argv) > 2 and sys.argv[1] else 'NVDA'

    # data processing
    df, spark = run_lakehouse(ticker=TICKER)


    # pandas df
    if df is None:
        df_local_file_path = os.path.join('data', 'df', TICKER, 'df.parquet')
        df = pd.read_parquet(df_local_file_path)

    df.info()


    # create target and input vals
    target_col = 'close'
    y = df.copy()[target_col].shift(-1) # avoid data leakage
    y = y.iloc[:-1] # drop the last row (y = nan)

    input_cols = [col for col in df.columns if col not in [target_col, 'dt']] # drop dt as year, month, date can capture sequence
    X = df.copy()[input_cols]
    X = X.iloc[:-1] # drop the last row

    # create train, val, test datasets
    from sklearn.model_selection import train_test_split
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1000, shuffle=False, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=1000, shuffle=False, random_state=42)


    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    cat_cols = ['year', 'month', 'date']
    num_cols = list(set(input_cols) - set(cat_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    spark.stop()
