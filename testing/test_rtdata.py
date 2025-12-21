# test_rtdata_behavior.py
import threading
from market_monitor.live_data_hub.real_time_data_hub import RTData


class TestRTDataThreadSafety:
    """Test per verificare comportamento attuale"""

    def test_concurrent_reads_writes(self):
        """Verifica letture/scritture concorrenti"""
        lock = threading.Lock()
        rtdata = RTData(lock, fields=["BID", "ASK"])
        rtdata.set_securities(["ISIN1", "ISIN2"])

        errors = []

        def writer():
            for i in range(100):
                try:
                    rtdata.update("ISIN1", {"BID": 100 + i, "ASK": 101 + i})
                except Exception as e:
                    errors.append(e)

        def reader():
            for i in range(100):
                try:
                    mid = rtdata.get_mid(["ISIN1", "ISIN2"])
                except Exception as e:
                    errors.append(e)

        # Lancia 2 writer e 3 reader
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errori durante concorrenza: {errors}"

    def test_get_mid_consistency(self):
        """Verifica che get_mid restituisca risultati consistenti"""
        lock = threading.Lock()
        rtdata = RTData(lock, fields=["BID", "ASK"], mid_key=["BID", "ASK"])
        rtdata.set_securities(["ISIN1"])

        rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})
        mid = rtdata.get_mid(["ISIN1"])

        assert mid["ISIN1"] == 101.0

    def test_get_mid_thread_safe(self):
        lock = threading.Lock()
        rtdata = RTData(lock, fields=["BID", "ASK"])
        rtdata.set_securities(["ISIN1"])
        rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})

        results = []

        def reader():
            for _ in range(1000):
                mid = rtdata.get_mid(["ISIN1"])
                results.append(mid["ISIN1"])

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Tutti i risultati devono essere 101.0
        assert all(r == 101.0 for r in results)

TestRTDataThreadSafety().test_concurrent_reads_writes()
TestRTDataThreadSafety().test_get_mid_thread_safe()
