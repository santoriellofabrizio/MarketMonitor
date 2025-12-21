"""
Test RTData refactored with separate Store classes
"""
import unittest
from threading import Lock
from datetime import datetime, timedelta

from market_monitor_fi.live_data_hub.RTData import RTData


class TestRTDataStores(unittest.TestCase):
    """Test RTData with new store architecture"""

    def setUp(self):
        """Setup test fixtures"""
        self.lock = Lock()
        self.rtdata = RTData(
            locker=self.lock,
            fields=["BID", "ASK"],
            mid_key=["BID", "ASK"]
        )

    def test_market_store_initialization(self):
        """Test market store initialization"""
        securities = ["ISIN1", "ISIN2", "ISIN3"]
        self.rtdata.set_securities(securities)

        # Check securities were set
        self.assertEqual(set(self.rtdata.securities), set(securities))

        # Check market data initialized
        market_data = self.rtdata.get_data_field()
        self.assertEqual(len(market_data), len(securities))
        self.assertListEqual(list(market_data.columns), ["BID", "ASK"])

    def test_market_data_update(self):
        """Test market data updates"""
        securities = ["ISIN1", "ISIN2"]
        self.rtdata.set_securities(securities)

        # Update data
        self.rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})
        self.rtdata.update("ISIN2", {"BID": 200.0, "ASK": 202.0})

        # Verify data
        bid_data = self.rtdata.get_data_field(field="BID")
        self.assertEqual(bid_data["ISIN1"], 100.0)
        self.assertEqual(bid_data["ISIN2"], 200.0)

    def test_subscription_store(self):
        """Test subscription store tracking"""
        securities = ["ISIN1", "ISIN2"]
        self.rtdata.set_securities(securities)

        # Update data (should mark last_update)
        self.rtdata.update("ISIN1", {"BID": 100.0})

        # Check last_update
        last_updates = self.rtdata.last_update
        self.assertIn("ISIN1", last_updates)
        self.assertIsInstance(last_updates["ISIN1"], datetime)

    def test_currency_store(self):
        """Test currency store"""
        securities = ["ISIN1", "ISIN2"]
        self.rtdata.set_securities(securities)

        # Set currency info
        self.rtdata.currency_information = {
            "ISIN1": "USD",
            "ISIN2": "EUR"
        }

        # Verify
        currency_info = self.rtdata.currency_information
        self.assertEqual(currency_info["ISIN1"], "USD")
        self.assertEqual(currency_info["ISIN2"], "EUR")

        # Check currencies in book
        currencies_in_book = self.rtdata.currencies_in_book
        self.assertIn("USD", currencies_in_book)
        self.assertIn("EUR", currencies_in_book)

    def test_tracking_store_threshold(self):
        """Test tracking store for threshold violations"""
        securities = ["ISIN1"]
        self.rtdata.set_securities(securities)
        self.rtdata._max_var_threshold = 0.05  # 5% threshold

        # Set initial price
        self.rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})

        # Try to update with large variation
        self.rtdata.update("ISIN1", {"BID": 150.0}, perform_check=True)

        # Check threshold exceeded
        threshold_exceeded = self.rtdata.threshold_exceeded_instr
        self.assertIn("ISIN1", threshold_exceeded)

    def test_currency_conversion_eur(self):
        """Test EUR conversion with currency store"""
        securities = ["ISIN1", "USD", "GBP"]
        self.rtdata.set_securities(securities)

        # Set currency info
        self.rtdata.currency_information = {"ISIN1": "USD"}

        # Set market data
        self.rtdata.update("USD", {"BID": 1.1, "ASK": 1.1})  # EUR/USD
        self.rtdata.update("GBP", {"BID": 0.85, "ASK": 0.85})  # EUR/GBP
        self.rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})  # In USD

        # Get mid in EUR
        mid_eur = self.rtdata.get_mid_eur()

        # ISIN1 should be converted: 101.0 / 1.1 ≈ 91.82
        self.assertAlmostEqual(mid_eur["ISIN1"], 101.0 / 1.1, places=2)

    def test_backward_compatibility_properties(self):
        """Test backward compatibility properties"""
        securities = ["ISIN1"]
        self.rtdata.set_securities(securities)

        # Update data
        self.rtdata.update("ISIN1", {"BID": 100.0})

        # Test backward compat properties
        self.assertIsInstance(self.rtdata.last_update, dict)
        self.assertIsInstance(self.rtdata.subscription_status, dict)
        self.assertIsInstance(self.rtdata.threshold_exceeded_instr, set)
        self.assertIsInstance(self.rtdata.missing_currency_instruments, set)
        self.assertIsInstance(self.rtdata.currencies_in_book, set)

    def test_missing_currency_tracking(self):
        """Test missing currency instrument tracking"""
        securities = ["ISIN1"]
        self.rtdata.set_securities(securities)

        # Set invalid currency
        self.rtdata.currency_information = {"ISIN1": "INVALID"}

        # Should default to EUR and mark as missing
        missing = self.rtdata.get_instruments_with_missing_ccy()
        self.assertIn("ISIN1", missing)

        # Currency should default to EUR
        self.assertEqual(self.rtdata.currency_information["ISIN1"], "EUR")


if __name__ == '__main__':
    unittest.main()
