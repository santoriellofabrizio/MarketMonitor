import logging
import os
import time
from datetime import datetime
import xlwings as xw

from market_monitor.publishers.GUIRedis import GUIRedis
from market_monitor.gui.HeartBeat.HeartBeatAbstract import HeartBeatDuma
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI


class QuotingCheck(StrategyUI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gui_redis = GUIRedis()
        self.heartbeat = None
        heartbeat_params = kwargs.pop('heartbeat')
        self.wb = self.open_wb(heartbeat_params.pop("path"))
        self.heartbeat_interval = int(heartbeat_params.pop("heartbeat_interval"))
        self.heartbeat = HeartBeatDuma(self.wb, self.heartbeat_interval, **heartbeat_params)
        self.gui_redis.export_static_data(HEARTBEAT_PID=os.getpid())

    def wait_for_book_initialization(self):
        while self.market_data.get_data_field("book","redis") is None:
            time.sleep(1)
        return True

    def on_market_data_setting(self):
        self.market_data.subscription_dict_redis = ["book",
                                                    "time_now"]

    def update_HF(self):
        secs_lag = self.get_live_data()
        if secs_lag > 2:
            logging.warning(f"RedisPublisher late of {secs_lag}s")
        elif secs_lag > 10:
            self.heartbeat.send_stop_signal()
            logging.warning("Sending Stopping signal...")
        else:
            self.heartbeat.send_heartbeat()
            logging.warning("Sending heartbeat...")

    def get_live_data(self):

        time_update = self.market_data.get_data_field(field="time_now", index_data="redis")
        secs = (datetime.now() - datetime.fromisoformat(time_update)).total_seconds()
        return secs

    def open_wb(self, path):
        while count := 0 < 5:
            try:
                for app in xw.apps:
                    for wb in app.books:
                        if os.path.basename(wb.name) == os.path.basename(path):
                            logging.info(f"Il file '{path}' è già aperto in un'istanza di Excel.")
                            try:
                                # Verifica che 'PIDHeartbeat' sia un nome definito e non una cella
                                return wb
                            except Exception as e:
                                logging.warning(f"Errore durante l'assegnazione del valore a 'PIDHeartbeat': {e}")

                break  # Breaks out of the loop if the file isn't found already open
            except Exception as e:
                logging.error(f"Errore nell'esaminare i workbook aperti: {e}")
                time.sleep(1)  # Attende un secondo prima di riprovare
                count += 1

    def on_stop(self):
        self.heartbeat.send_stop_signal()
