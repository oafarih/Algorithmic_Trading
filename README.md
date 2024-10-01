
# Algorithmic Trading

This repository contains Python scripts for developing, backtesting, optimizing, and live trading of algorithmic strategies. The project allows for extensive analysis of strategies, including walk-forward and portfolio optimization.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
- [Results](#results)
- [Live Trading Setup](#live-trading-setup)
- [Future Work](#future-work)
- [License](#license)

## Features

- **Backtesting**: Simulate trading strategies on historical data to evaluate performance.
- **Strategy & Portfolio Optimization**: Optimize strategy parameters and allocate assets efficiently.
- **Walk-Forward Optimization**: Test strategies across multiple time periods to avoid overfitting.
- **Live Trading**: Execute strategies in real-time with connection to Binance API.
- **Technical Indicators**: Leverage indicators like moving averages, RSI, and more.
- **Machine Learning Models**: Use ML models for advanced predictions in certain strategies.

## Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/oafarih/Algorithmic_Trading.git
cd Algorithmic_Trading
pip install -r requirements.txt
```

## Usage

1. **Configure API Keys**: Set up Binance API keys in a `binance_keys.py` file:
   ```python
   api_key = 'your_api_key'
   secret_key = 'your_secret_key'
   ```

2. **Download OHLCV data**:
   - Use the `_store_data.py` script to test and optimize strategies:
   ```bash
   python _store_data.py
   ```

3. **Backtest and Optimize a Strategy**:
   - Use the `backtest.py` script to test and optimize strategies:
   ```python
   class Strategy:
      # Limit orders strategy
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

      # Entry and Exit signals
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

   def main_ma_cross(asset: str):

      print(f'Backtesting {asset}...')
      data = OHLCVData(asset).time_bars('hour', 1)
      bt_kwargs = {'data': data, 'strategy': 'moving_average_crossover', 'market_side': 'long', 'limit_orders': True}
      opt_kwargs = {'optimizer': 'gbrt', 'n_calls': 110, 'n_initial_points': 10, 'initial_point_generator': 'random',
                  'acq_func': 'EI', 'xi': 0.01, 'kappa': 1.2, 'verbose': False, 'n_jobs': 1}
      
      space = [
         Integer(5, 50, name='period_fast'),
         Integer(20, 100, name='period_slow'),
         Categorical(['SMA', 'EMA', 'HMA'], name='ma_type'),
         Real(1.0, 5.0, name='sl_multiplier'),
         Real(1.0, 5.0, name='tp_multiplier'),
         Categorical(['atr'], name='limit_order_method')
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

   if __name__ == '__main__':

      assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC', 'DOT', 'UNI', 'FTM', 'LINK', 'XRP']
      for asset in assets:
         main_ma_cross(asset)
   ```
   - This will backtest a strategy, optimize parameters, and perform walk-forward optimization.

4. **Live Trading**:
   - After a strategy and portfolio are found, use the `_bot.py` and `_bot_optimize.py` for live trading.
   - Set up the virtual machine environment and cron jobs to automate live execution.

## Configuration

- **Data Sources**: The repository uses Binance API for both historical and real-time data. Ensure API keys are properly configured.
- **Virtual Machine**: Set up a virtual machine to run live trading strategies 24/7. Automate execution using cron jobs.

## Scripts Overview

- **`backtest.py`**: Backtest strategies, optimize parameters, and perform portfolio optimization and walk-forward analysis.
- **`_bot_optimize.py`**: Optimize and test strategies for live trading.
- **`_bot.py`**: Execute trading strategies in real-time using live market data.
- **`_store_data.py`**: Fetch and store historical market data from Binance.
- **`technical_analysis.py`**: Generate technical indicators to support strategy development.

## Results

Sample backtest results can be found in the provided Jupyter notebook:
- **`_results.ipynb`**: Backtest and optimization results of a moving average crossover strategy and of applying machine learning models to market data.

## Live Trading Setup

To set up for live trading:
1. Deploy the code on a **Virtual Machine (VM)** to ensure continuous operation.
2. Set up **cron jobs** to automate tasks such as updating data, running optimizations, and executing trades in real-time.

Example cron job setup for running the bot every day at midnight, running optimizations every month and updating data every hour:
```bash
0 0 * * * /usr/bin/python3 /path/to/Algorithmic_Trading/_bot.py
0 0 1 * * /usr/bin/python3 /path/to/Algorithmic_Trading/_bot_optimize.py
0 * * * * /usr/bin/python3 /path/to/Algorithmic_Trading/_store_data.py
```

## Future Work

- **Add More Strategies**: Implement advanced trading strategies, including pairs trading and volatility arbitrage.
- **Machine Learning Expansion**: Further explore machine learning-based strategies and integrate predictive models.
- **Real-Time Monitoring**: Develop dashboards or alert systems to monitor live trading activity.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
