import pandas as pd
from datetime import datetime, timedelta

from backtest import Strategy
from binance_keys import api_key, secret_key
from _bot_optimize import round_decimals_down

from binance import Client

def getdata(asset, tf, dt_start):
    client = Client(api_key, secret_key)
    df = pd.DataFrame(client.get_historical_klines(asset+'USDT', tf, dt_start))
    df = df.iloc[:-1, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.loc[:, 'Time'] = pd.to_datetime(df.Time, unit='ms')
    df.set_index('Time', inplace=True)
    df = df.astype(float)
    df.attrs['name'] = asset

    return df

def trade_signal_ma_crossover(asset):
    
    position_frame = pd.read_csv('position_frame.csv')
    position_asset = position_frame[position_frame['Asset'] == asset]
    args = [arg.strip() for arg in position_asset['Args'].item()[1:-1].split(',')]
    args = [int(x) if x.isdigit() else float(x) if '.' in x else bool(x) if x in ['True', 'False'] else x for x in args]

    dt_start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    data = getdata(asset, '1h', dt_start)
    st = getattr(Strategy(data, market_side='long', limit_orders=False), 'moving_average_crossover')
    signals = st(*args)
    
    return signals, data


def trade_signal_ml(asset):

    position_frame = pd.read_csv('position_frame.csv')
    position_asset = position_frame[position_frame['Asset'] == asset]
    args = [arg.strip() for arg in position_asset['Args'].item()[1:-1].split(',')]
    args = [int(x) if x.isdigit() else float(x) if '.' in x else bool(x) if x in ['True', 'False'] else x for x in args]
    args[-1] = False

    dt_start = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')
    data = getdata(asset, '4h', dt_start)
    st = getattr(Strategy(data, market_side='long', limit_orders=False), 'returns_sign_ml')
    signals = st(*args)
    
    return signals, data

def trade_spot(asset, commission=0.001):
    client = Client(api_key, secret_key)
    # https://www.binance.com/en/trade-rule
    decimals_quantity = {'BTC': 5, 'ETH': 4, 'SOL': 3, 'ADA': 1, 'MATIC': 1, 
                         'DOT': 2, 'UNI': 2, 'FTM': 0, 'LINK': 2, 'XRP': 0}
    position_frame = pd.read_csv('position_frame.csv')
    position_asset = position_frame[position_frame['Asset'] == asset]

    signals, data = trade_signal_ml(asset)

    if (signals.iloc[-1] == 1) & (position_asset['Position'].item() == 0):
        qty = round_decimals_down(position_asset['Cash'] / data['Close'].iloc[-1], decimals_quantity[asset])
        order = client.create_order(symbol=asset+'USDT', side='BUY', type='MARKET', quantity=qty)
        
        buy_price = float(order['fills'][-1]['price'])

        position_frame.loc[position_frame['Asset'] == asset, 'Position'] = 1
        position_frame.loc[position_frame['Asset'] == asset, 'Price'] = buy_price

        print(f'Buy {asset} @ {buy_price}')
    
    elif (signals.iloc[-1] == -1) & (position_asset['Position'].item() == 1):
        qty = round_decimals_down(float(client.get_asset_balance(asset)['free']), decimals_quantity[asset])
        order = client.create_order(symbol=asset+'USDT', side='SELL', type='MARKET', quantity=qty)
        
        sell_price = float(order['fills'][0]['price'])
        buy_price = position_frame.loc[position_frame['Asset'] == asset, 'Price'].item()
        returns = (sell_price / buy_price) - 1 - commission

        position_frame.loc[position_frame['Asset'] == asset, 'Position'] = 0
        position_frame.loc[position_frame['Asset'] == asset, 'Cash'] = position_asset['Cash'] * (1 + returns)

        print(f'Sell {asset} @ {sell_price} - Profit: {returns*100:.2f}%')

    position_frame.to_csv('position_frame.csv', index=False)

if __name__ == '__main__':
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'DOT', 'UNI', 'FTM', 'LINK', 'XRP']
    for asset in assets:
        trade_spot(asset)