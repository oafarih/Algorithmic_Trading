import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from pypfopt import expected_returns, risk_models, black_litterman, efficient_frontier, cla, hierarchical_portfolio
from technical_analysis import TechnicalAnalysis, MovingAverage, OHLCVData, TimeSeriesSplitWalkForwardDT, TimeSeriesSplitPurge

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
import pickle
import os

class Strategy:

    def __init__(self, data: pd.DataFrame, market_side: str = 'long', limit_orders: bool = False) -> pd.Series:
        self.data = data
        self.market_side = market_side
        self.limit_orders = limit_orders
        self.ta = TechnicalAnalysis(self.data)
        self.ma = MovingAverage(self.data)

    def limit_orders_func(self, sl_multiplier: float = 1.0, tp_multiplier: float = 1.0, limit_order_method: str = 'atr') -> pd.DataFrame:
        
        df_limit_orders = self.data.copy()
        if limit_order_method == 'atr':
            df_limit_orders['ATR'] = self.ta.ATR()
            if self.market_side == 'long':
                df_limit_orders['StopLoss'] = df_limit_orders['Close'] - (df_limit_orders['ATR'] * sl_multiplier)
                df_limit_orders['TakeProfit'] = df_limit_orders['Close'] + (df_limit_orders['ATR'] * tp_multiplier)
            elif self.market_side == 'short':
                df_limit_orders['StopLoss'] = df_limit_orders['Close'] + (df_limit_orders['ATR'] * sl_multiplier)
                df_limit_orders['TakeProfit'] = df_limit_orders['Close'] - (df_limit_orders['ATR'] * tp_multiplier)
        
        return df_limit_orders.loc[:, ['StopLoss', 'TakeProfit']]

    def moving_average_crossover(self, period_fast: int = 20, period_slow: int = 50, ma_type: str = 'EMA', *args) -> pd.Series | tuple:
        
        if self.limit_orders:
            if args[0] > args[1] or period_fast >= period_slow:
                if self.market_side in ['long', 'short']:
                    return pd.Series(0, index=self.data.index), self.limit_orders_func(*args)
                else:
                    return pd.DataFrame({'long': 0, 'short': 0}, index=self.data.index), self.limit_orders_func(*args)
        if period_fast >= period_slow:
            if self.market_side in ['long', 'short']: 
                return pd.Series(0, index=self.data.index)
            else:
                return pd.DataFrame({'long': 0, 'short': 0}, index=self.data.index)
        
        self.data['MA_FAST'] = getattr(self.ma, ma_type)(period=period_fast)
        self.data['MA_SLOW'] = getattr(self.ma, ma_type)(period=period_slow)

        long_signal = (self.data['MA_FAST'] > self.data['MA_SLOW']) & (self.data['MA_FAST'].shift(1) < self.data['MA_SLOW'].shift(1))
        short_signal = (self.data['MA_FAST'] < self.data['MA_SLOW']) & (self.data['MA_FAST'].shift(1) > self.data['MA_SLOW'].shift(1))

        long_exit = (self.data['Close'] < self.data['MA_SLOW'])
        short_exit = (self.data['Close'] > self.data['MA_SLOW'])
        
        if self.market_side == 'long':
            if self.limit_orders:
                signals = pd.Series(np.where(long_signal, 1, 0), index=self.data.index)
                return signals, self.limit_orders_func(*args)
            else:
                signals = pd.Series(np.select([long_signal, long_exit], [1, -1], default=0), index=self.data.index)
                return signals

        elif self.market_side == 'short':
            if self.limit_orders:
                signals = pd.Series(np.where(short_signal, -1, 0), index=self.data.index)
                return signals, self.limit_orders_func(*args)
            else:
                signals = pd.Series(np.select([short_signal, short_exit], [1, -1], default=0), index=self.data.index)
                return signals
            
        else:
            if self.limit_orders:
                signals_long = pd.Series(np.where(long_signal, 1, 0), name='long', index=self.data.index)
                signals_short = pd.Series(np.where(short_signal, -1, 0), name='short', index=self.data.index)
                signals = pd.concat([signals_long, signals_short], axis=1)
                return signals, self.limit_orders_func(*args)
            else:
                signals_long = pd.Series(np.select([long_signal, long_exit], [1, -1], default=0), name='long', index=self.data.index)
                signals_short = pd.Series(np.select([short_signal, short_exit], [1, -1], default=0), name='short', index=self.data.index)
                signals = pd.concat([signals_long, signals_short], axis=1)
        
        return signals
    
    def returns_sign_ml(self, period_trend: int = 12, period_season: int = 12, nlags: int = 10, alpha: float = 1.0, save_model: bool = True) -> pd.Series | tuple:
        
        def season_grouper(data, n):
            div, rest = divmod(len(data), n)
            idxs = [list(range(1, n+1))] * div
            if rest:
                idxs += [list(range(1, rest+1))]
            idxs = np.concatenate(idxs)
            return idxs
        
        def get_corr(arr, nlags):
            coefs = []
            for lag in range(1, nlags+1):
                ylag = arr.shift(lag).dropna()
                ytrue = arr.loc[ylag.index]
                coef, _ = pearsonr(ylag, ytrue)
                coefs.append(coef)
            corr = pd.Series(dict(zip(range(1, nlags+1), coefs)))
            return corr
        
        def get_corr_dual(arr1, arr2, nlags):
            coefs = []
            for lag in range(1, nlags+1):
                ylag = arr2.shift(lag).dropna()
                ytrue = arr1.loc[ylag.index]
                coef, _ = pearsonr(ylag, ytrue)
                coefs.append(coef)
            corr = pd.Series(dict(zip(range(1, nlags+1), coefs)))
            return corr

        returns = self.ta.RETURN(period=1)
        trend = self.ma.ZLEMA(arr=returns, period=period_trend)
        residuals = returns - trend
        seasonal = residuals.groupby(season_grouper(self.data, period_season)).ewm(period_season).mean().reset_index(level=0, drop=True).sort_index()
        residuals = residuals - seasonal

        df_trend = pd.DataFrame({'trend': trend, 'seasonal': seasonal, 'residuals': residuals, 'returns': returns}).fillna(0)
        asset = self.data.attrs['name']

        if save_model:
            corr_returns_trend = get_corr_dual(df_trend['returns'], df_trend['trend'], nlags=nlags**2)
            corr_returns_seasonal = get_corr_dual(df_trend['returns'], df_trend['seasonal'], nlags=nlags**2)
            corr_trend_seasonal = get_corr_dual(df_trend['trend'], df_trend['seasonal'], nlags=nlags**2)
            corr_returns = get_corr(df_trend['returns'], nlags=nlags**2)
            corr_trend = get_corr(df_trend['trend'], nlags=nlags**2)
            corr_seasonal = get_corr(df_trend['seasonal'], nlags=nlags**2)

            lags_in = list(
                set(
                corr_returns_trend.abs().nlargest(nlags).index.tolist() +\
                    corr_returns_seasonal.abs().nlargest(nlags).index.tolist() +\
                        corr_trend_seasonal.abs().nlargest(nlags).index.tolist() +\
                            corr_returns.abs().nlargest(nlags).index.tolist() +\
                                corr_trend.abs().nlargest(nlags).index.tolist() +\
                                    corr_seasonal.abs().nlargest(nlags).index.tolist()
                    )
                )
            
            for lag in np.sort(lags_in):

                df_trend[f'trend_LAG_{lag}'] = df_trend['trend'].shift(lag)
                df_trend[f'seasonal_LAG_{lag}'] = df_trend['seasonal'].shift(lag)
                df_trend[f'residuals_LAG_{lag}'] = df_trend['residuals'].shift(lag)
                df_trend[f'returns_LAG_{lag}'] = df_trend['returns'].shift(lag)

            df_trend['target'] = df_trend['returns'].shift(-1)
            df_trend.fillna(0, inplace=True)

            X_trend = df_trend.drop(['target'], axis=1)
            y_trend = df_trend['target']

            purged_cv = TimeSeriesSplitPurge(df=X_trend, n_splits=5)
            y_preds = []
            for train_idx, test_idx in purged_cv:
                X_train, X_test = X_trend.iloc[train_idx], X_trend.iloc[test_idx]
                y_train, y_test = y_trend.iloc[train_idx], y_trend.iloc[test_idx]

                model = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
                model.fit(X_train, y_train)
                
                y_pred = pd.Series(model.predict(X_test), index=y_test.index)
                y_preds.append(y_pred)
            
            y_pred = pd.concat(y_preds, axis=0).sort_index(ascending=True)

            if len(y_pred) < len(y_trend):
                y_pred_new = pd.Series(0.0, index=y_trend.index)
                y_pred_new.loc[y_pred.index] = y_pred
                y_pred = y_pred_new

            pipeline = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha))])
            pipeline.fit(X_trend, y_trend)

            if not os.path.exists(f'{asset}_models'):
                os.mkdir(f'{asset}_models')
            with open(f'{asset}_models/pipeline_{period_trend}_{period_season}_{nlags}_{alpha}.pkl', 'wb') as f:
                pickle.dump(pipeline, f)
        else:
            with open(f'{asset}_models/pipeline_{period_trend}_{period_season}_{nlags}_{alpha}.pkl', 'rb') as f:
                pipeline = pickle.load(f)
                for file in os.listdir(f'{asset}_models'):
                    if os.path.isfile(f'{asset}_models/{file}') and file != f'pipeline_{period_trend}_{period_season}_{nlags}_{alpha}.pkl':
                        os.remove(f'{asset}_models/{file}')
            
            lags_in = [col for col in pipeline.feature_names_in_ if 'LAG' in col]
            lags_in = list(set([int(lag.split('_')[-1]) for lag in lags_in]))

            for lag in np.sort(lags_in):

                df_trend[f'trend_LAG_{lag}'] = df_trend['trend'].shift(lag)
                df_trend[f'seasonal_LAG_{lag}'] = df_trend['seasonal'].shift(lag)
                df_trend[f'residuals_LAG_{lag}'] = df_trend['residuals'].shift(lag)
                df_trend[f'returns_LAG_{lag}'] = df_trend['returns'].shift(lag)

            df_trend.fillna(0, inplace=True)
            X_trend = df_trend
            
            y_pred = pipeline.predict(X_trend)

        long_signal = y_pred > 0
        short_signal = y_pred < 0

        long_exit = y_pred <= 0
        short_exit = y_pred >= 0

        if self.market_side == 'long':
            signals = pd.Series(np.select([long_signal, long_exit], [1, -1], default=0), index=self.data.index)
            return signals
        
        elif self.market_side == 'short':
            signals = pd.Series(np.select([short_signal, short_exit], [1, -1], default=0), index=self.data.index)
            return signals
        
        else:
            signals_long = pd.Series(np.select([long_signal, long_exit], [1, -1], default=0), name='long', index=self.data.index)
            signals_short = pd.Series(np.select([short_signal, short_exit], [1, -1], default=0), name='short', index=self.data.index)
            signals = pd.concat([signals_long, signals_short], axis=1)
            return signals
        
class Backtest:
    
    def __init__(self, data: pd.DataFrame, strategy: str, market_side: str = 'long', limit_orders: bool = False):
        self.data = data
        self.market_side = market_side
        self.limit_orders = limit_orders
        self.strategy = strategy
        self.strategy_func = getattr(Strategy(data=self.data, market_side=self.market_side, limit_orders=self.limit_orders), self.strategy)
        self.ml_strategies = ['returns_sign_ml']

    def bt_trades(self, signals: pd.Series | pd.DataFrame | tuple, commission: float = 0.001) -> pd.DataFrame:
        
        if isinstance(signals, tuple):
            signals, df_limit_orders = signals

        if self.market_side in ['long', 'short']:
            if signals[signals != 0].shape[0] == 0:
                return pd.DataFrame()
            df_bt = self.data.loc[signals.index[0]:signals.index[-1]].copy()
            df_bt['Signal'] = signals
        else:
            if (signals['long']+signals['short']).loc[lambda x: x != 0].shape[0] == 0:
                return pd.DataFrame()
            df_bt = self.data.loc[signals.index[0]:signals.index[-1]].copy()
            df_bt['Signal_long'] = signals['long']
            df_bt['Signal_short'] = signals['short']
        
        if self.limit_orders:
            dt_last_exit_time = df_bt.index[0] - pd.Timedelta(days=1)
            df_bt = pd.concat([df_bt, df_limit_orders], axis=1)
            _signals = df_bt.loc[df_bt['Signal'] != 0, 'Signal']
            df_trades = df_bt.loc[_signals.index]

            if df_trades.shape[0] == 0:
                return pd.DataFrame()

            trades = []
            for idx, row in df_trades.iterrows():
                if idx > dt_last_exit_time:
                    trade = {'EntryTime': idx, 'EntryPrice': row['Close']}
                    stop_loss, take_profit = row['StopLoss'], row['TakeProfit']

                    if self.market_side == 'long':
                        idx_limit_order_trigger = (df_bt.index > idx) & ((df_bt['Low'] < stop_loss) | (df_bt['High'] > take_profit))
                    elif self.market_side == 'short':
                        idx_limit_order_trigger = (df_bt.index > idx) & ((df_bt['High'] > stop_loss) | (df_bt['Low'] < take_profit))

                    if idx_limit_order_trigger.sum() == 0:
                        trade['ExitTime'] = df_bt.index[-1]
                        trade['ExitPrice'] = df_bt['Close'].iloc[-1]
                        if self.market_side == 'long':
                            trade['ReturnPct'] = (trade['ExitPrice'] - trade['EntryPrice']) / trade['EntryPrice']                
                        elif self.market_side == 'short':
                            trade['ReturnPct'] = (trade['EntryPrice'] - trade['ExitPrice']) / trade['EntryPrice']
                        trades.append(trade)
                        break

                    row_limit_order = df_bt.loc[idx_limit_order_trigger].iloc[0]
                    dt_last_exit_time = row_limit_order.name
                    trade['ExitTime'] = row_limit_order.name

                    if self.market_side == 'long':
                        if row_limit_order['Low'] < stop_loss:
                            trade['ExitPrice'] = stop_loss
                        elif row_limit_order['High'] > take_profit:
                            trade['ExitPrice'] = take_profit
                        trade['ReturnPct'] = (trade['ExitPrice'] - trade['EntryPrice']) / trade['EntryPrice']
                    
                    elif self.market_side == 'short':
                        if row_limit_order['High'] > stop_loss:
                            trade['ExitPrice'] = stop_loss
                        elif row_limit_order['Low'] < take_profit:
                            trade['ExitPrice'] = take_profit
                        trade['ReturnPct'] = (trade['EntryPrice'] - trade['ExitPrice']) / trade['EntryPrice']
                
                    trades.append(trade)

            trades = pd.DataFrame(trades)
            
        else:
            if self.market_side in ['long', 'short']:
                _signals = df_bt.loc[df_bt['Signal'] != 0, 'Signal']
                _signals = _signals.diff().fillna(_signals.iloc[0]).loc[lambda x: x != 0].apply(np.sign)
                df_trades = df_bt.loc[_signals.index]

                if df_trades.shape[0] == 0:
                    return pd.DataFrame()

                if self.market_side == 'long':
                    open_signal, close_signal = 1, -1
                elif self.market_side == 'short':
                    open_signal, close_signal = -1, 1
                
                if df_trades['Signal'].iloc[0] == close_signal:
                    df_trades = df_trades.iloc[1:]
                    if df_trades.shape[0] == 0:
                        return pd.DataFrame()
                if df_trades['Signal'].iloc[-1] == open_signal:
                    last_tick = df_bt.iloc[-1]
                    last_tick['Signal'] = close_signal
                    df_trades = pd.concat([df_trades, last_tick.to_frame().T], axis=0)

                df_open = df_trades.loc[df_trades['Signal'] == open_signal]
                df_close = df_trades.loc[df_trades['Signal'] == close_signal]

                trades = pd.DataFrame({'EntryTime': df_open.index.tolist(), 
                                    'EntryPrice': df_open['Close'].to_list(), 
                                    'ExitTime': df_close.index.tolist(), 
                                    'ExitPrice': df_close['Close'].to_list()})
                
                if self.market_side == 'long':
                    trades['ReturnPct'] = (trades['ExitPrice'] - trades['EntryPrice']) / trades['EntryPrice']                
                elif self.market_side == 'short':
                    trades['ReturnPct'] = (trades['EntryPrice'] - trades['ExitPrice']) / trades['EntryPrice']

            else:
                _signals_long = df_bt.loc[df_bt['Signal_long'] != 0, 'Signal_long']
                _signals_short = df_bt.loc[df_bt['Signal_short'] != 0, 'Signal_short']
                _signals_long = _signals_long.diff().fillna(_signals_long.iloc[0]).loc[lambda x: x != 0].apply(np.sign)
                _signals_short = _signals_short.diff().fillna(_signals_short.iloc[0]).loc[lambda x: x != 0].apply(np.sign)
                df_trades_long = df_bt.loc[_signals_long.index].rename(columns={'Signal_long': 'Signal'})
                df_trades_short = df_bt.loc[_signals_short.index].rename(columns={'Signal_short': 'Signal'})

                results = []
                for market_side, df_trades in zip(['long', 'short'], [df_trades_long, df_trades_short]):
                    if df_trades.shape[0] == 0:
                        continue
                
                    if market_side == 'long':
                        open_signal, close_signal = 1, -1
                    elif market_side == 'short':
                        open_signal, close_signal = -1, 1

                    if df_trades['Signal'].iloc[0] == close_signal:
                        df_trades = df_trades.iloc[1:]
                        if df_trades.shape[0] == 0:
                            continue
                    if df_trades['Signal'].iloc[-1] == open_signal:
                        last_tick = df_bt.iloc[-1]
                        last_tick['Signal'] = close_signal
                        df_trades = pd.concat([df_trades, last_tick.to_frame().T], axis=0)

                    df_open = df_trades.loc[df_trades['Signal'] == open_signal]
                    df_close = df_trades.loc[df_trades['Signal'] == close_signal]

                    trades = pd.DataFrame({'EntryTime': df_open.index.tolist(), 
                                            'EntryPrice': df_open['Close'].to_list(), 
                                            'ExitTime': df_close.index.tolist(), 
                                            'ExitPrice': df_close['Close'].to_list()})

                    if market_side == 'long':
                        trades['ReturnPct'] = (trades['ExitPrice'] - trades['EntryPrice']) / trades['EntryPrice']                
                    elif market_side == 'short':
                        trades['ReturnPct'] = (trades['EntryPrice'] - trades['ExitPrice']) / trades['EntryPrice']

                    results.append(trades)

                if len(results) == 0:
                    return pd.DataFrame()
                
                def remove_overlapping_trades(trades: pd.DataFrame) -> pd.DataFrame:
                    trades = trades.sort_values('EntryTime').reset_index(drop=True)
                    if trades.loc[trades['ExitTime'].shift() > trades['EntryTime']].shape[0] > 0:
                        idx_drop = trades.loc[trades['ExitTime'].shift() > trades['EntryTime']].index
                        trades = trades.drop(idx_drop).reset_index(drop=True)
                        remove_overlapping_trades(trades)
                    return trades
                
                trades = pd.concat(results)
                trades = remove_overlapping_trades(trades)
            
        trades['ReturnPct'] = trades['ReturnPct'].fillna(0) - commission
        return trades

    def bt_stats(self, trades: pd.DataFrame, trading_days: int = 365, risk_free_rate: float = 0.03) -> pd.Series:
        
        if len(trades) == 0:
            return pd.Series()
        
        trades['EquityCurve'] = (trades['ReturnPct'] + 1).cumprod() - 1

        start, end = trades['EntryTime'].iloc[0], trades['ExitTime'].iloc[-1]
        duration = end - start
        duration_in_market = (trades['ExitTime'] - trades['EntryTime']).sum()
        market_exposure_time = duration_in_market / duration

        def td_format(td_object):
            seconds = int(td_object.total_seconds())
            periods = [
                ('Y', 60*60*24*365),
                ('M', 60*60*24*30),
                ('W', 60*60*24*7),
                ('D', 60*60*24),
                ('H', 60*60),
                #('Min', 60),
                #('S', 1)
            ]

            strings=[]
            for period_name, period_seconds in periods:
                if seconds >= period_seconds:
                    period_value , seconds = divmod(seconds, period_seconds)
                    strings.append(f'{period_value}{period_name}')

            return ' '.join(strings)

        duration_str = td_format(duration.to_pytimedelta())

        return_final = trades['EquityCurve'].iloc[-1]
        returns_total = trades['ReturnPct'].sum()
        buy_hold_returns = (self.data.loc[end, 'Close'] / self.data.loc[start, 'Close']) - 1

        n_trades = trades['ReturnPct'].count()
        win_rate = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].count() / n_trades

        best_return = trades['ReturnPct'].max()
        worst_return = trades['ReturnPct'].min()
        avg_return = trades['ReturnPct'].mean()
        volatility = trades['ReturnPct'].std() if trades['ReturnPct'].count() > 1 else 0

        avg_positive_return = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].mean() if trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].count() > 0 else 0
        avg_negative_return = trades.loc[trades['ReturnPct'] < 0, 'ReturnPct'].mean() if trades.loc[trades['ReturnPct'] < 0, 'ReturnPct'].count() > 0 else 0

        return_annualized = ((1 + return_final) ** trading_days) - 1 if duration.days == 0 else ((1 + return_final) ** (trading_days / duration.days)) - 1
        volatility_annualized = volatility * np.sqrt(trading_days)

        drawdown = (trades['EquityCurve'] / trades['EquityCurve'].cummax()) - 1
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        if trades.loc[trades['EquityCurve'] < 0, 'EquityCurve'].count() > 0:
            duration_downside = (trades.loc[trades['EquityCurve'] < 0, 'ExitTime'] - trades.loc[trades['EquityCurve'] < 0, 'EntryTime']).sum()
            downside_exposure_time = duration_downside / duration_in_market
        else:
            downside_exposure_time = 0.0

        if trades.loc[trades['ReturnPct'] < 0, 'ReturnPct'].count() > 0:
            profit_factor = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].sum() / trades.loc[trades['ReturnPct'] < 0, 'ReturnPct'].abs().sum()
            win_loss_ratio = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].count() / trades.loc[trades['ReturnPct'] < 0, 'ReturnPct'].count()
        else:
            profit_factor = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].sum() / 1e-2
            win_loss_ratio = trades.loc[trades['ReturnPct'] > 0, 'ReturnPct'].count() / 1

        sharpe_ratio = (return_annualized - risk_free_rate) / volatility_annualized if volatility_annualized else (return_annualized - risk_free_rate) / 1e-2
        sqn_score = np.sqrt(n_trades) * avg_return / volatility if volatility else np.sqrt(n_trades) * avg_return / 1e-2

        def consecutive_win_loss(returns: pd.Series) -> tuple:
            consecutive_wins = pd.Series(0, index=returns.index)
            consecutive_losses = pd.Series(0, index=returns.index)
            wins_final, losses_final = [], []

            for i, ret in enumerate(returns):
                if i > 0:
                    if ret > 0:
                        if returns.iloc[i-1] < 0: 
                            losses_final.append(consecutive_losses.iloc[i-1])
                        consecutive_wins.iloc[i] = consecutive_wins.iloc[i-1] + 1
                        consecutive_losses.iloc[i] = 0
                    elif ret < 0:
                        if returns.iloc[i-1] > 0: 
                            wins_final.append(consecutive_wins.iloc[i-1])
                        consecutive_losses.iloc[i] = consecutive_losses.iloc[i-1] + 1
                        consecutive_wins.iloc[i] = 0

            max_consecutive_wins = consecutive_wins.max()
            max_consecutive_losses = consecutive_losses.max()
            avg_consecutive_wins = 0 if len(wins_final) == 0 else np.mean(wins_final)
            avg_consecutive_losses = 0 if len(losses_final) == 0 else np.mean(losses_final)

            return avg_consecutive_wins, avg_consecutive_losses, max_consecutive_wins, max_consecutive_losses
        
        avg_consecutive_wins, avg_consecutive_losses, max_consecutive_wins, max_consecutive_losses = consecutive_win_loss(trades['ReturnPct'])
        
        stats_names = ['Start', 'End', 'Duration', 'Exposure Time', 'Downside Exposure Time', 'Returns', 'Total Returns', 'Buy & Hold Returns', 'Number Of Trades', 
                       'Win Rate', 'Best Return', 'Worst Return', 'Avg Return', 'Avg Positive Return', 'Avg Negative Return', 'Volatility', 
                       'Returns Annual', 'Volatility Annual', 'Max Drawdown', 'Avg Drawdown', 'Profit Factor', 'Win Loss Ratio', 
                       'Sharpe Ratio', 'SQN', 'Avg Consecutive Wins', 'Avg Consecutive Losses', 'Max Consecutive Wins', 'Max Consecutive Losses']
        
        stats = [start, end, duration_str, market_exposure_time, downside_exposure_time, return_final, returns_total, buy_hold_returns, n_trades, 
                 win_rate, best_return, worst_return, avg_return, avg_positive_return, avg_negative_return, volatility, 
                 return_annualized, volatility_annualized, max_drawdown, avg_drawdown, profit_factor, win_loss_ratio, 
                 sharpe_ratio, sqn_score, avg_consecutive_wins, avg_consecutive_losses, max_consecutive_wins, max_consecutive_losses]
        
        return pd.Series({k:v for k,v in zip(stats_names, stats)})

    def bt_run(self, *args) -> pd.DataFrame:
    
        signals = self.strategy_func(*args)
        trades = self.bt_trades(signals)
        stats = self.bt_stats(trades)
        return trades, stats

    def bt_optimize(self, objective: str, space: list, optimizer: str = 'gbrt', **kwargs) -> tuple:
        
        optimizers = {'gp':gp_minimize, 'forest':forest_minimize, 'gbrt':gbrt_minimize, 'dummy':dummy_minimize}
        opt = optimizers[optimizer]

        def opt_objective_func(args: list) -> float:
            signals = self.strategy_func(*args)
            trades = self.bt_trades(signals)
            opt_value = self.opt_objective(objective, trades)
            return -opt_value
        
        optimizer_result = opt(func=opt_objective_func, dimensions=space, **kwargs)
        
        args = optimizer_result.x
        signals = self.strategy_func(*args)
        trades = self.bt_trades(signals)
        stats = self.bt_stats(trades)
        stats['Args'] = args
        return trades, stats
    
    def bt_optimize_walk_forward(self, objective: str, space: list, optimizer: str = 'gbrt', timeframe: str = 'month', n_train: int = 4, n_test: int = 1, anchored: bool = False, **kwargs) -> tuple:
        
        optimizers = {'gp':gp_minimize, 'forest':forest_minimize, 'gbrt':gbrt_minimize, 'dummy':dummy_minimize}
        opt = optimizers[optimizer]

        dt_split = TimeSeriesSplitWalkForwardDT(self.data, timeframe, n_train, n_test, anchored)
        wf_trades, wf_stats = [], []
        
        for train_idx, test_idx in dt_split:
            
            def opt_objective_func(args: list):
                strategy_func = getattr(Strategy(data=self.data.loc[train_idx], market_side=self.market_side, limit_orders=self.limit_orders), self.strategy)
                signals = strategy_func(*args)
                trades = self.bt_trades(signals)
                opt_value = self.opt_objective(objective, trades)
                return -opt_value
            
            optimizer_result = opt(func=opt_objective_func, dimensions=space, **kwargs)
            
            strategy_func = getattr(Strategy(data=self.data.loc[train_idx[0]:test_idx[-1]], market_side=self.market_side, limit_orders=self.limit_orders), self.strategy)
            args = optimizer_result.x
            if self.strategy in self.ml_strategies:
                args[-1] = False
            signals = strategy_func(*args)
            signals = (signals[0][test_idx], signals[1].loc[test_idx]) if self.limit_orders else (signals[test_idx] if self.market_side in ['long', 'short'] else signals.loc[test_idx])
            trades = self.bt_trades(signals)
            stats = self.bt_stats(trades)
            stats['Args'] = args

            wf_trades.append(trades)
            wf_stats.append(stats)
        
        trades = pd.concat(wf_trades, axis=0).reset_index(drop=True) if len(wf_trades) > 0 else pd.DataFrame()
        stats = self.bt_stats(trades)
        stats_wf = pd.concat(wf_stats, axis=1).T if len(wf_stats) > 0 else pd.DataFrame()
        
        return trades, stats, stats_wf
    
    def opt_objective(self, objective: str, trades: pd.Series) -> float:
        if len(trades) == 0: 
            return -1e6
        stats = self.bt_stats(trades)

        if objective in stats.index:
            value = stats[objective]
        
        elif objective == 'Returns_AvgDrawdown':
            if stats['Returns Annual'] <= 0:
                value = -1e6
            elif np.abs(stats['Avg Drawdown']) < 1e-1:
                value = stats['Returns Annual'] / 1e-1
            else:
                value = stats['Returns Annual'] / np.abs(stats['Avg Drawdown'])
        
        elif objective == 'SharpeRatio_AvgDrawdown':
            if stats['Sharpe Ratio'] < 0:
                value = -1e6
            elif np.abs(stats['Avg Drawdown']) < 1e-1:
                value = stats['Sharpe Ratio'] / 1e-1
            else:
                value = stats['Sharpe Ratio'] / np.abs(stats['Avg Drawdown'])

        elif objective == 'Custom_Objective':
            if (stats['Profit Factor'] >= 1) & (stats['Win Loss Ratio'] >= 1) & (stats['Sharpe Ratio'] >= 1): 
                value = stats['Returns'] * stats['Profit Factor'] * stats['Win Loss Ratio'] * stats['Sharpe Ratio']
            else:
                value = stats['Returns']
        
        if str(value) == 'nan': return -1e6

        return value

class Portfolio:

    def __init__(self, returns: pd.DataFrame, frequency: int = 365, returns_method: str = 'mean_historical_return', risk_method: str = 'exp_cov', **kwargs):
        self.returns = returns
        self.assets_curve = (1 + self.returns.fillna(0)).cumprod() - 1
        self.assets_returns = self.assets_curve.iloc[-1]
        self.mu = expected_returns.return_model(self.returns, frequency=frequency, method=returns_method, returns_data=True, **kwargs)
        self.cov = risk_models.risk_matrix(self.returns, frequency=frequency, method=risk_method, returns_data=True, **kwargs)
    
    def black_litterman_allocation(self, view: dict, optimize: str = 'sharpe_ratio') -> tuple:
        bl = black_litterman.BlackLittermanModel(self.cov, absolute_views=view)
        ef = efficient_frontier.EfficientFrontier(bl.bl_returns(), bl.bl_cov())

        if optimize == 'sharpe_ratio':
            weights_bl = pd.Series(ef.max_sharpe())
        elif optimize == 'volatility':
            weights_bl = pd.Series(ef.min_volatility())

        bl_expected_return, bl_volatility, bl_sharpe_ratio = ef.portfolio_performance()
        bl_metrics = pd.Series([bl_expected_return, bl_volatility, bl_sharpe_ratio], index=['Expected Annual Return', 'Annual Volatility', 'Sharpe ratio'])

        bl_returns = self.assets_returns * weights_bl
        bl_results = pd.DataFrame([weights_bl, bl_returns], index=['Weights', 'Returns']).T
        bl_results = bl_results[bl_results['Weights'] > 0.005].sort_values(by='Weights', ascending=False)

        bl_cumulative_returns = (self.assets_curve.loc[:, bl_results.index] * bl_results['Weights']).sum(axis=1)
        self.assets_curve['BL_Portfolio'] = bl_cumulative_returns

        return bl_metrics, bl_results
    
    def critical_line_algorithm(self, max_weight: float, min_weight: float = 0.0, optimize: str = 'sharpe_ratio') -> tuple:
        cl_algo = cla.CLA(self.mu, self.cov, weight_bounds=(min_weight, max_weight))

        if optimize == 'sharpe_ratio':
            weights_cla = pd.Series(cl_algo.max_sharpe())
        elif optimize == 'volatility':
            weights_cla = pd.Series(cl_algo.min_volatility())

        cla_expected_return, cla_volatility, cla_sharpe_ratio = cl_algo.portfolio_performance()
        cla_metrics = pd.Series([cla_expected_return, cla_volatility, cla_sharpe_ratio], index=['Expected Annual Return', 'Annual Volatility', 'Sharpe ratio'])

        returns_cla = self.assets_returns * weights_cla
        cla_results = pd.DataFrame([weights_cla, returns_cla], index=['Weights', 'Returns']).T
        cla_results = cla_results[cla_results['Weights'] > 0.005].sort_values(by='Weights', ascending=False)
        
        cla_cumulative_returns = (self.assets_curve.loc[:, cla_results.index] * cla_results['Weights']).sum(axis=1)
        self.assets_curve['CLA_Portfolio'] = cla_cumulative_returns

        return cla_metrics, cla_results
    
    def hierarchical_risk_parity(self) -> tuple:
        hrp = hierarchical_portfolio.HRPOpt(self.returns, self.cov)

        weights_hrp = pd.Series(hrp.optimize())

        hrp_expected_return, hrp_volatility, hrp_sharpe_ratio = hrp.portfolio_performance()
        hrp_metrics = pd.Series([hrp_expected_return, hrp_volatility, hrp_sharpe_ratio], index=['Expected Annual Return', 'Annual Volatility', 'Sharpe ratio'])

        returns_hrp = self.assets_returns * weights_hrp
        hrp_results = pd.DataFrame([weights_hrp, returns_hrp], index=['Weights', 'Returns']).T
        hrp_results = hrp_results[hrp_results['Weights'] > 0.005].sort_values(by='Weights', ascending=False)
        
        hrp_cumulative_returns = (self.assets_curve.loc[:, hrp_results.index] *  hrp_results['Weights']).sum(axis=1)
        self.assets_curve['HRP_Portfolio'] = hrp_cumulative_returns

        return hrp_metrics, hrp_results
    
    def plot_equity_curve(self, model: str = 'CLA', log_scale: bool = True) -> None:

        eq_curve = self.assets_curve[f'{model}_Portfolio']
        if log_scale:
            eq_curve = eq_curve.apply(lambda x: np.log(1+x))

        plt.figure(figsize=(12, 8))
        plt.plot(eq_curve, label=f'{model} Portfolio Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid()  
        
    def plot_weights(self, weights: pd.Series, model: str = 'CLA') -> None:

        plt.figure(figsize=(12, 8))
        plt.pie(weights, labels=weights.index.tolist(), autopct='%1.1f%%')
        plt.title(f'{model} Portfolio Weights')

def main_ma_cross(asset: str):

    print(f'Backtesting {asset}...')
    data = OHLCVData(asset).time_bars('hour', 1)
    bt_kwargs = {'data': data, 'strategy': 'moving_average_crossover', 'market_side': 'long', 'limit_orders': False}
    opt_kwargs = {'optimizer': 'gbrt', 'n_calls': 110, 'n_initial_points': 10, 'initial_point_generator': 'random',
                  'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.2, 'verbose': False, 'n_jobs': 1}
    
    space = [
        Integer(5, 50, name='period_fast'),
        Integer(20, 100, name='period_slow'),
        Categorical(['SMA', 'EMA', 'HMA'], name='ma_type'),
        #Real(1.0, 10.0, name='sl_multiplier'),
        #Real(1.0, 10.0, name='tp_multiplier'),
        #Categorical(['atr'], name='limit_order_method')
        ]
    
    objective = 'Returns_AvgDrawdown'
    timeframe, n_train, n_test, anchored = 'month', 12, 3, True
    bt = Backtest(**bt_kwargs)
    trades, stats, stats_wf = bt.bt_optimize_walk_forward(objective=objective, space=space, timeframe=timeframe, n_train=n_train, n_test=n_test, anchored=anchored, **opt_kwargs)
    stats.name = asset
    
    if not '_results_ma_crossover' in os.listdir():
        os.mkdir('_results_ma_crossover')
    if not asset in os.listdir('_results_ma_crossover'):
        os.mkdir(f'_results_ma_crossover/{asset}')

    trades.to_csv(f'_results_ma_crossover/{asset}/trades.csv', index=False)
    stats_wf.to_csv(f'_results_ma_crossover/{asset}/stats.csv', index=False)
    stats.to_csv(f'_results_ma_crossover/{asset}/performance.csv')
    print(f'{asset} done!')

def main_ml(asset: str):

    print(f'Backtesting {asset}...')
    data = OHLCVData(asset).time_bars('hour', 4)
    bt_kwargs = {'data': data, 'strategy': 'returns_sign_ml', 'market_side': 'long', 'limit_orders': False}
    opt_kwargs = {'optimizer': 'gbrt', 'n_calls': 110, 'n_initial_points': 10, 'initial_point_generator': 'random',
                  'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.2, 'verbose': False, 'n_jobs': 1}
    
    space = [
        Integer(4, 24, name='period_trend'),
        Integer(2, 12, name='period_season'),
        Integer(5, 10, name='nlags'),
        Real(1.0, 500.0, name='alpha'),
        Categorical([True], name='save_model')
        ]
    
    objective = 'Custom_Objective'
    timeframe, n_train, n_test, anchored = 'week', 52, 1, True
    bt = Backtest(**bt_kwargs)
    trades, stats, stats_wf = bt.bt_optimize_walk_forward(objective=objective, space=space, timeframe=timeframe, n_train=n_train, n_test=n_test, anchored=anchored, **opt_kwargs)
    stats.name = asset
    
    if not '_results_ml' in os.listdir():
        os.mkdir('_results_ml')
    if not asset in os.listdir('_results_ml'):
        os.mkdir(f'_results_ml/{asset}')

    trades.to_csv(f'_results_ml/{asset}/trades.csv', index=False)
    stats_wf.to_csv(f'_results_ml/{asset}/stats.csv', index=False)
    stats.to_csv(f'_results_ml/{asset}/performance.csv')
    print(f'{asset} done!')

if __name__ == '__main__':
    import multiprocessing

    main_func = main_ml # main_ma_cross

    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'DOT', 'UNI', 'FTM', 'LINK', 'XRP']
        
    multiprocessing.freeze_support()
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    args = [(asset,) for asset in assets]
    results = pool.starmap(main_func, args)
    pool.close()
    pool.join()






