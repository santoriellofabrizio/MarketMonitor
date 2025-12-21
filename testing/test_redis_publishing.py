"""
Test RedisPublisher publishing with MockPricePublisher.
Uses IntegratedStrategyTestRunner with trades disabled.
"""
import logging

from market_monitor_fi.strategy.UserStrategy.StrategyRegister import register_strategy
from testing.IntegratedStrategyTestRunner import run_integrated_test
from testing.TestStrategy.MockPricePublisher import MockPricePublisher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Register strategy
    register_strategy("MockPricePublisher", MockPricePublisher)
    
    # Run test
    print("\n╔" + "="*68 + "╗")
    print("║       REDIS STORE ROUTING TEST - MockPricePublisher             ║")
    print("╚" + "="*68 + "╝\n")
    
    results = run_integrated_test(
        strategy_name="MockPricePublisher",
        gui_type="RedisMessaging",
        duration_seconds=30,
        num_etf_instruments=5,
        market_update_interval=0.5,
        activate_trades=False,  # Disable trade_distributor
        activate_bloomberg_mock=True,
    )
    
    print(f"\nTest result: {'PASS' if results['success'] else 'FAIL'}")
