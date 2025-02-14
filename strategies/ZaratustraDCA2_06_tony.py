import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class Zara(IStrategy):
    """
        Personalized Trading Strategy with Risk Management, DCA, and Technical Indicators

        This automated strategy combines advanced risk management techniques with Dollar Cost Averaging (DCA)
        to optimize market operations. The strategy employs technical indicators (e.g., ADX, PDI, MDI) to define entry
        and exit conditions for both long and short positions. It also includes dynamic stoploss logic based on ATR and
        protections like cooldown periods and loss guards.

        Key features:
        - Long and short trading enabled with dynamic position adjustments.
        - Protections against overtrading using cooldown mechanisms.
        - Use of technical indicators to track market trends and trigger signals.
        - Supports dynamic stake adjustments based on successful trade entries.

        NOTE: This strategy is under development, and some functions may be disabled or incomplete. Parameters like ROI
        targets and advanced stoploss logic are subject to future optimization. Test thoroughly before using it in live trading.
    """

    ### Strategy parameters ###
    leverage_factor = 8
    exit_profit_only = True
    ignore_roi_if_entry_signal = True
    can_short = True
    use_exit_signal = True
    #stoploss = -0.02 * leverage_factor
    startup_candle_count: int = 100
    timeframe = '1m'
    stoploss = -0.05 # * leverage_factor
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True  # Default - not necessary for this examp
    edge = False
    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    # max_dca_multiplier = 1  # Maximum DCA multiplier
    # max_dca_multiplier = 1 + max_entry_position_adjustment + 0.5 * max_entry_position_adjustment * (max_entry_position_adjustment + 1) / 2
    max_dca_multiplier = 3.5  # Maximum DCA multiplier

    # ROI table:
    minimal_roi = {}

    ### Hyperopt ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 120, default=72, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=False, space="protection", optimize=True)

    ### Protections ###
    @property
    def protections(self):
        """
            Defines the protections to apply during trading operations.
        """

        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    ### Dollar Cost Averaging (DCA) ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically based on the DCA multiplier.
            - The proposed stake is divided by the maximum DCA multiplier (`self.max_dca_multiplier`)
              to determine the adjusted stake.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """

        # Calculates the adjusted stake amount based on the DCA multiplier.
        # adjusted_stake = proposed_stake / self.max_dca_multiplier
        #
        # # Automatically adjusts to the minimum stake if it is too low.
        # if adjusted_stake < min_stake:
        #     adjusted_stake = min_stake
        # return adjusted_stake

        if self.config['stake_amount'] == 'unlimited':
            custom_stake_amount = proposed_stake
        else:
            custom_stake_amount = self.wallets.get_total_stake_amount() / self.config['max_open_trades']
        return custom_stake_amount / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.
        """

        no_lev_c_p = current_profit / self.leverage_factor
        if trade.entry_side == "buy":  # For longs
            if no_lev_c_p > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if no_lev_c_p > -0.04 and trade.nr_of_successful_entries == 1:
                return None
            if no_lev_c_p > -0.06 and trade.nr_of_successful_entries == 2:
                return None
            if no_lev_c_p > -0.08 and trade.nr_of_successful_entries == 3:
                return None

        else:  # For shorts
            if no_lev_c_p > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if no_lev_c_p > -0.04 and trade.nr_of_successful_entries == 1:
                return None
            if no_lev_c_p > -0.06 and trade.nr_of_successful_entries == 2:
                return None
            if no_lev_c_p > -0.08 and trade.nr_of_successful_entries == 3:
                return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        try:
            stake_amount = filled_entries[0].cost
            stake_amount /= self.leverage_factor

            if count_of_entries > 1:
                stake_amount = stake_amount * (1 + (count_of_entries - 1) * 0.5)  # 50% increase per additional entry

            return stake_amount

        except Exception as exception:
            logger.error(f"Error adjusting DCA position for the pair {trade.pair}: {exception}")
            return None

#    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
#                        current_profit: float,
#                        **kwargs) -> float:
#        """
#            Implements a dynamic stoploss based on ATR (Average True Range):
#            - Calculates the ATR from the latest analyzed dataframe.
#            - Returns a stoploss dynamically set as a percentage of the current rate.
#        """

#        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
#        atr = dataframe['atr'].iloc[-1]
#        return -atr / current_rate

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['dx']  = ta.SMA(ta.DX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['adx'] = ta.SMA(ta.ADX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['pdi'] = ta.SMA(ta.PLUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['mdi'] = ta.SMA(ta.MINUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines the conditions for long/short entries based on
            technical indicators such as ADX, PDI, and MDI.
        """

        df.loc[
            (
                    (qtpylib.crossed_above(df['dx'], df['pdi'])) &
                    (df['adx'] > df['mdi']) &
                    (df['pdi'] > df['mdi'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Long')

        df.loc[
            (
                    (qtpylib.crossed_above(df['dx'], df['mdi'])) &
                    (df['adx'] > df['pdi']) &
                    (df['mdi'] > df['pdi'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Short')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines exit conditions for trades based on the ADX indicator:
            - Exit Long: Triggered when 'dx' crosses below 'adx' and ADX is strong (>25).
            - Exit Short: Triggered when 'dx' crosses below 'adx' and ADX is weak (≤25).
        """

        df.loc[
            (
                    (qtpylib.crossed_below(df['dx'], df['adx'])) &
                    (df['adx'] > 25)  # Strong ADX (above 25, to confirm trend)

            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Long')

        df.loc[
            (
                    (qtpylib.crossed_below(df['dx'], df['adx'])) &
                    (df['adx'] <= 25)  # Weak ADX (below 25, lacks trend strength)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Short')

        return df

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return self.leverage_factor
