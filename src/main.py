import data_handling


if __name__ == '__main__':
    TICKER = 'NVDA'

    # extract
    stock_price_data = data_handling.extract_daily_stock_data(ticker=TICKER)

    # load
    data_handling.load_to_s3(data=stock_price_data, ticker=TICKER)

    # bronze
    data_handling.bronze.store_data_to_local(data=stock_price_data, ticker=TICKER)
    bronze_df = data_handling.bronze.create_df(data=stock_price_data, ticker=TICKER)
    bronze_s3_path = data_handling.bronze.load_to_s3(df=bronze_df, ticker=TICKER)

    # start spark session
    spark = data_handling.config_and_start_spark_session()

    # silver - retrieve delta table in the bronze layer
    bronze_delta_talbe = data_handling.retrieve_delta_table(spark=spark, s3_path=bronze_s3_path)
    silver_df = data_handling.silver.process(delta_table=bronze_delta_talbe)
    silver_s3_path = data_handling.silver.load(df=silver_df, ticker=TICKER)

    # gold
    silver_delta_table = data_handling.retrieve_delta_table(spark=spark, s3_path=silver_s3_path)
    gold_df = data_handling.gold.process(delta_table=silver_delta_table)
    gold_s3_path = data_handling.gold.load(df=gold_df, ticker=TICKER)

    if gold_s3_path: spark.stop()
