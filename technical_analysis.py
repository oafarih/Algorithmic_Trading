import pandas as pd
import numpy as np

class OHLCVData:

    def __init__(self, asset: str):
        self.asset = asset
        self.df = pd.read_parquet(f'_data/{self.asset}USDT.parquet', engine='pyarrow').ffill()
        self.df.attrs['name'] = self.asset

    def time_bars(self, timeframe: str = 'minute', freq: int = 1) -> pd.DataFrame:
        '''
        Time Bars

        Parameters
        ----------
        timeframe : str
            The timeframe to resample the data.
        freq : int
            The frequency to resample the data.
        heiken_ashi : bool
            The Heiken Ashi candlestick chart.
        
        Returns
        -------
        data : pd.DataFrame
            The resampled data based on the given timeframe.
        '''
        tfs = {'year': 'YS', 'quarter': 'QS', 'week': 'W-MON', 'month': 'MS', 'day': 'D', 'hour': 'h', 'minute': 'min'}
        if timeframe == 'minute' and freq == 1:
            return self.df
        tf = f'{freq}{tfs[timeframe]}'
        data = self.df.resample(tf, closed='left', label='left').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        return data
    
class MovingAverage:

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def SMA(self, arr: str | pd.Series = 'Close', period: int = 10) -> pd.Series:
        '''
        Simple Moving Average

        Parameters
        ----------
        arr : str | pd.Series
            The input array to calculate the moving average.
        period : int
            The period of the moving average.

        Returns
        -------
        series : pd.Series
            The moving average of the input array.
        '''
        if isinstance(arr, str):
            arr = self.df[arr]
        series = arr.copy()
        return series.rolling(period).mean()
    
    def EMA(self, arr: str | pd.Series = 'Close', period: int = 10) -> pd.Series:
        '''
        Exponential Moving Average

        Parameters
        ----------
        arr : str | pd.Series
            The input array to calculate the moving average.
        period : int
            The period of the moving average.

        Returns
        -------
        series : pd.Series
            The moving average of the input array.
        '''
        if isinstance(arr, str):
            arr = self.df[arr]
        series = arr.copy()
        return series.ewm(span=period, adjust=False).mean()
    
    def WMA(self, arr: str | pd.Series = 'Close', period: int = 10) -> pd.Series:
        '''
        Weighted Moving Average

        Parameters
        ----------
        arr : str | pd.Series
            The input array to calculate the moving average.
        period : int
            The period of the moving average.

        Returns
        -------
        series : pd.Series
            The moving average of the input array.
        '''
        if isinstance(arr, str):
            arr = self.df[arr]
        series = arr.copy()
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    
    def HMA(self, arr: str | pd.Series = 'Close', period: int = 10) -> pd.Series:
        '''
        Hull Moving Average

        Parameters
        ----------
        arr : str | pd.Series
            The input array to calculate the moving average.
        period : int
            The period of the moving average.

        Returns
        -------
        series : pd.Series
            The moving average of the input array.
        '''
        if isinstance(arr, str):
            arr = self.df[arr]
        series = arr.copy()
        half_period = int(period/2)
        sqrt_period = int(np.sqrt(period))
        return self.WMA(2*self.WMA(series, half_period) - self.WMA(series, period), sqrt_period)
    
    def ZLEMA(self, arr: str | pd.Series = 'Close', period: int = 10) -> pd.Series:
        '''
        Zero Lag Exponential Moving Average

        Parameters
        ----------
        arr : str | pd.Series
            The input array to calculate the moving average.
        period : int
            The period of the moving average.

        Returns
        -------
        series : pd.Series
            The moving average of the input array.
        '''
        if isinstance(arr, str):
            arr = self.df[arr]
        series = arr.copy()
        return self.EMA((series + series.diff()), period)
    
class TechnicalAnalysis:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ma = MovingAverage(df)
        
    def RSI(self, period: int = 10, ma_type: str = 'EMA') -> pd.Series:
        '''
        Relative Strength Index

        Parameters
        ----------
        period : int
            The period of the moving average.
        ma_type : str
            The type of moving average to use.

        Returns
        -------
        series : pd.Series
            The relative strength index.
        '''
        delta = self.df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up.loc[up < 0] = 0
        down.loc[down > 0] = 0

        gain = getattr(self.ma, ma_type)(up, period)
        loss = getattr(self.ma, ma_type)(down.abs(), period)
        rs = gain / loss

        series = 100 - (100 / (1 + rs))
        series = series.astype('float')
        series.name = 'RSI'
        return series
    
    def STOCH(self, period: int = 10, ma_type: str = 'EMA') -> pd.Series:
        '''
        Stochastic Oscillator

        Parameters
        ----------
        period : int
            The period of the moving average.
        ma_type : str
            The type of moving average to use.

        Returns
        -------
        series : pd.Series
            The stochastic oscillator.
        '''
        lowest_low = self.df['Close'].rolling(period).min()
        highest_high = self.df['Close'].rolling(period).max()

        stoch = 100 * (self.df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)

        series = getattr(self.ma, ma_type)(stoch, period)
        series = series.astype('float')
        series.name = 'STOCH'
        return series

    def MACD(self, period_fast : int = 10, period_slow: int = 20, ma_type: str = 'EMA') -> pd.DataFrame:
        '''
        Moving Average Convergence Divergence

        Parameters
        ----------
        period_fast : int
            The period of the fast moving average.
        period_slow : int
            The period of the slow moving average.
        ma_type : str
            The type of moving average to use.

        Returns
        -------
        dataframe : pd.DataFrame
            The moving average convergence divergence.
        '''
        ma_fast = getattr(self.ma, ma_type)(self.df['Close'], period_fast)
        ma_slow = getattr(self.ma, ma_type)(self.df['Close'], period_slow)
        macd = ma_fast - ma_slow
        macd_signal = macd - getattr(self.ma, ma_type)(macd, period_fast)
        dataframe = pd.concat([macd, macd_signal], axis=1)
        dataframe = dataframe.astype('float')
        dataframe.columns = ['MACD', 'MACD_SIGNAL']
        return dataframe

    def TR(self) -> pd.Series:
        '''
        True Range

        Returns
        -------
        series : pd.Series
            The true range.
        '''
        series = self.df['High'] - self.df['Low']
        series = series.astype('float')
        series.name = 'TR'
        return series
    
    def ATR(self, period: int = 10, ma_type: str = 'EMA') -> pd.Series:
        '''
        Average True Range

        Parameters
        ----------
        period : int
            The period of the moving average.
        ma_type : str
            The type of moving average to use.

        Returns
        -------
        series : pd.Series
            The average true range.
        '''
        tr = self.TR()
        series = getattr(self.ma, ma_type)(tr, period)
        series = series.astype('float')
        series.name = 'ATR'
        return series
    
    def RETURN(self, period: int = 1, log: bool = True) -> pd.Series:
        '''
        Return

        Parameters
        ----------
        period : int
            The period of the return.
        log : bool
            If True, the return will be log.

        Returns
        -------
        series : pd.Series
            The return.
        '''
        series = self.df['Close'].pct_change(period)
        series = series.astype('float')
        series.name = 'RETURN'
        if log:
            series = np.log(1 + series)
            series.name = 'RETURN'
        return series
    
def TimeSeriesSplitPurge(df, n_splits=5):
    assert(n_splits > 2)
    n = df.shape[0]
    step_size = n // n_splits
    pct_purging = 1/(n_splits*8)
    n_purging = round(n*pct_purging)

    for split in range(n_splits):
        if split > 0 and split < n_splits - 1:
            start_train_1 = 0
            end_train_1 = (split * step_size) - n_purging
            start_test = split * step_size
            end_test = (split + 1) * step_size
            start_train_2 = ((split + 1) * step_size) + n_purging
            end_train_2 = n

            idx_train_1 = np.arange(start_train_1, end_train_1)
            idx_train_2 = np.arange(start_train_2, end_train_2)
            idx_train = np.concatenate([idx_train_1, idx_train_2])
            idx_test = np.arange(start_test, end_test)
        else:
            if split == 0:
                start_test = 0
                end_test = step_size
                start_train = step_size + n_purging
                end_train = n

                idx_train = np.arange(start_train, end_train)
                idx_test = np.arange(start_test, end_test)

            elif split == n_splits - 1:
                start_train = 0
                end_train = n - step_size - n_purging
                start_test = n - step_size
                end_test = n 

                idx_train = np.arange(start_train, end_train)
                idx_test = np.arange(start_test, end_test)

        yield(idx_train, idx_test)

def TimeSeriesSplitWalkForwardDT(df, tf='month', n_train=4, n_test=1, anchored=False):
    df_freq_start = {'year': 'YS', 'month': 'MS', 'week': 'W-MON', 'day': 'D', 'hour': 'h', 'minute': 'min'}
    df_freq_end = {'year': 'YE', 'month': 'ME', 'week': 'W-SUN', 'day': 'D', 'hour': 'h', 'minute': 'min'}

    dt_range_start = pd.date_range(start=df.index[0], end=df.index[-1], freq=df_freq_start[tf], normalize=True)
    dt_range_end = pd.date_range(start=df.index[0], end=df.index[-1], freq=df_freq_end[tf], normalize=True)

    dt_range_start = dt_range_start[(dt_range_start >= df.index[0]) & (dt_range_start <= df.index[-1])]
    dt_range_end = dt_range_end[(dt_range_end >= df.index[0]) & (dt_range_end <= df.index[-1])]

    dt_range_start = dt_range_start[:-1]
    if tf in ['year', 'month', 'week']:
        dt_range_end += pd.Timedelta(days = 1, minutes = -1)
        if df.index[0] != dt_range_start[0]:
            dt_range_end = dt_range_end[1:]
    elif tf in ['day', 'hour', 'minute']:
        dt_range_end += pd.Timedelta(minutes = -1)
        dt_range_end = dt_range_end[1:]
    
    dt_range = list(zip(dt_range_start, dt_range_end))
    
    for idx in range(n_train-1, len(dt_range)-n_test, n_test):
        train_start = df.index[0] if anchored else dt_range[idx - (n_train - 1)][0]
        train_end = dt_range[idx][1]
        test_start = dt_range[idx + 1][0]
        test_end = dt_range[idx + n_test][1]

        train_idx = df.index[(df.index >= train_start) & (df.index <= train_end)]
        test_idx = df.index[(df.index >= test_start) & (df.index <= test_end)]
        
        yield(train_idx, test_idx)