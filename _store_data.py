import pandas as pd
from binance import Client
from datetime import timedelta
import os

from binance_keys import api_key, secret_key

client = Client(api_key, secret_key)
directory = '_data'
if not directory in os.listdir():
    os.mkdir(directory)

def get_data(asset, start='2017-01-01'):

    df = pd.DataFrame(client.get_historical_klines(asset+'USDT', '1m', start))
    df = df.iloc[:-1, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.astype(float)
    return df

def store_data(asset):

    if asset in os.listdir(directory):
        df = pd.read_parquet(f'{directory}/{asset}USDT.parquet')
        ts = df.index[-1].date()
        df = df[:str(ts - timedelta(days=1))]
        data = get_data(asset, start=str(ts))
        data_1min = pd.concat([df, data], axis=0)
        data_1min.to_parquet(f'{directory}/{asset}USDT.parquet')
    else:
        data_1min = get_data(asset)
        data_1min.to_parquet(f'{directory}/{asset}USDT.parquet')

if __name__ == '__main__':
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'DOT', 'UNI', 'FTM', 'LINK', 'XRP']
    for asset in assets:
        store_data(asset)