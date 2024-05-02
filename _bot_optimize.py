import pandas as pd

from backtest import Backtest, Portfolio
from technical_analysis import OHLCVData
from binance_keys import api_key, secret_key

from binance import Client
from skopt.space import Real, Integer, Categorical

import math
import os

def optimize_strategy_(timeframe, freq, asset, strategy, space, market_side='long', limit_orders=False, objective='Returns', **kwargs):

    data = OHLCVData(asset).time_bars(timeframe, freq)
    bt_kwargs = {'data': data, 'strategy': strategy, 'market_side': market_side, 'limit_orders': limit_orders}

    bt = Backtest(**bt_kwargs)
    trades, stats = bt.bt_optimize(objective=objective, space=space, **kwargs)

    return trades, stats

def optimize_portfolio_(returns, frequency=365, max_weight=0.2, min_weight=0.03, optimize='sharpe_ratio', **kwargs):

    pf = Portfolio(returns, frequency=frequency, **kwargs)
    metrics, weights = pf.critical_line_algorithm(max_weight = max_weight, min_weight = min_weight, optimize = optimize)

    return metrics, weights

def round_decimals_down(number, decimals):
    
    if decimals == 0:
        return int(math.floor(number))

    factor = 10 ** decimals
    return float(math.floor(number * factor) / factor)

def close_positions_(commission=0.001):
    client = Client(api_key, secret_key)
    decimals_quantity = {'BTC': 5, 'ETH': 4, 'SOL': 3, 'ADA': 1, 'MATIC': 1, 
                         'DOT': 2, 'UNI': 2, 'FTM': 0, 'LINK': 2, 'XRP': 0}
    position_frame = pd.read_csv('position_frame.csv')
    if position_frame['Position'].sum() > 0:
        open_positions = position_frame[position_frame['Position'] == 1]

        for position in open_positions.index:
            asset = position_frame.loc[position, 'Asset'] 
            qty = round_decimals_down(float(client.get_asset_balance(asset)['free']), decimals_quantity[asset])
            order = client.create_order(symbol=asset+'USDT', side='SELL', type='MARKET', quantity=qty)

            sell_price = float(order['fills'][0]['price'])
            buy_price = position_frame.loc[position, 'Price'].item()
            returns = (sell_price / buy_price) - 1 - commission

            position_frame.loc[position, 'Position'] = 0
            position_frame.loc[position, 'Cash'] = position_frame.loc[position, 'Cash'] * (1 + returns)

            print(f'Sell {asset} @ {sell_price} - Profit: {returns*100:.2f}%')
    
        position_frame.to_csv('position_frame.csv', index=False)

def update_position_frame_(weights):
    if not 'position_frame.csv' in os.listdir():
        cash = 1000
    else:
        close_positions_()
        position_frame = pd.read_csv('position_frame.csv')
        cash = position_frame['Cash'].sum()
    
    position_frame = weights.reset_index().drop(columns=['Returns'], axis=1)
    position_frame.columns = ['Asset', 'Weights', 'Args']
    position_frame['Position'] = 0
    position_frame['Price'] = 0
    position_frame['Cash'] = position_frame['Weights'] * cash

    position_frame.to_csv('position_frame.csv', index=False)

def main_ma_crossover(assets):
    
        strategy = 'moving_average_crossover'
        market_side = 'long'
        limit_orders = False
        objective = 'Custom_Objective'
        space = [
            Integer(5, 100, name='period_fast'),
            Integer(10, 200, name='period_slow'),
            Categorical(['SMA', 'EMA', 'HMA'], name='ma_type'),
            #Real(1.0, 20.0, name='sl_multiplier'),
            #Real(1.0, 20.0, name='tp_multiplier'),
            #Categorical(['atr'], name='limit_order_method')
        ]
        opt_kwargs = {'optimizer': 'gbrt', 'n_calls': 110, 'n_initial_points': 10, 'initial_point_generator': 'random',
                    'acq_func': 'EI', 'xi': 0.05, 'kappa': 1.2, 'verbose': True, 'n_jobs': -1}
        
        freq, frequency = 'W', 52
        returns = []
        performance = []
        for asset in assets:
            trades, stats = optimize_strategy_('hour', 1, asset, strategy, space, market_side, limit_orders, objective, **opt_kwargs)
            trades.index = pd.to_datetime(trades['ExitTime'].tolist())
            asset_returns = trades.groupby(pd.Grouper(freq=freq))['ReturnPct'].apply(lambda x: (x + 1).prod() - 1)
            asset_returns.name = asset
            stats.name = asset
            returns.append(asset_returns)
            performance.append(stats)
    
        returns = pd.concat(returns, axis=1)
        performance = pd.concat(performance, axis=1).T
    
        max_weight, min_weight, optimize = 0.2, 0.03, 'sharpe_ratio'
        metrics, weights = optimize_portfolio_(returns, frequency, max_weight, min_weight, optimize=optimize)
        weights = pd.concat([weights, performance.loc[weights.index, 'Args']], axis=1)
        update_position_frame_(weights)
    
        return performance, metrics

def main_ml(assets):

    strategy = 'returns_sign_ml'
    market_side = 'long'
    limit_orders = False
    objective = 'Custom_Objective'
    space = [
        Integer(4, 24, name='period_trend'),
        Integer(2, 12, name='period_season'),
        Integer(5, 10, name='nlags'),
        Real(1.0, 500.0, name='alpha'),
        Categorical([True], name='save_model')
    ]
    opt_kwargs = {'optimizer': 'gbrt', 'n_calls': 110, 'n_initial_points': 10, 'initial_point_generator': 'random',
                  'acq_func': 'EI', 'xi': 0.05, 'kappa': 1.2, 'verbose': True, 'n_jobs': -1}
    
    freq, frequency = 'D', 365
    returns = []
    performance = []
    for asset in assets:
        trades, stats = optimize_strategy_('hour', 4, asset, strategy, space, market_side, limit_orders, objective, **opt_kwargs)
        trades.index = pd.to_datetime(trades['ExitTime'].tolist())
        asset_returns = trades.groupby(pd.Grouper(freq=freq))['ReturnPct'].apply(lambda x: (x + 1).prod() - 1)
        asset_returns.name = asset
        stats.name = asset
        returns.append(asset_returns)
        performance.append(stats)

    returns = pd.concat(returns, axis=1)
    performance = pd.concat(performance, axis=1).T

    max_weight, min_weight, optimize = 0.2, 0.03, 'sharpe_ratio'
    metrics, weights = optimize_portfolio_(returns, frequency, max_weight, min_weight, optimize=optimize)
    weights = pd.concat([weights, performance.loc[weights.index, 'Args']], axis=1)
    update_position_frame_(weights)

    return performance, metrics

if __name__ == '__main__':
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'DOT', 'UNI', 'FTM', 'LINK', 'XRP']
    performance, metrics = main_ml(assets)
    performance.to_csv('performance.csv')
    metrics.to_csv('metrics.csv')