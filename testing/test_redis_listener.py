"""
Test RedisPublisher listener: MockPricePublisher → RedisPublisher → MockRedisListener
"""
import logging
import time
import threading

from market_monitor_fi.strategy.UserStrategy.StrategyRegister import register_strategy
from testing.TestStrategy.MockPricePublisher import MockPricePublisher
from testing.TestStrategy.MockRedisListener import MockRedisListener
from testing.IntegratedStrategyTestRunner import IntegratedStrategyTestRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    print("\n╔" + "="*68 + "╗")
    print("║       REDIS LISTENER TEST - Publisher → RedisPublisher → Listener        ║")
    print("╚" + "="*68 + "╝\n")
    
    # Register strategies
    register_strategy("MockPricePublisher", MockPricePublisher)
    register_strategy("MockRedisListener", MockRedisListener)
    
    # 1. Start Publisher
    logger.info("Starting MockPricePublisher...")
    publisher_runner = IntegratedStrategyTestRunner(
        strategy_name="MockPricePublisher",
        gui_type="RedisMessaging",
        duration_seconds=60,
        activate_trades=False,
        activate_bloomberg_mock=True,
        activate_redis=False,  # Publisher doesn't need RedisPublisher input
        market_update_interval=1.0
    )
    
    if not publisher_runner.setup():
        logger.error("Publisher setup failed")
        exit(1)
    
    # Start publisher
    for thread in publisher_runner.threads:
        thread.start()
    
    if publisher_runner.bloomberg_mock:
        publisher_runner.bloomberg_mock.start()
    
    pub_thread = threading.Thread(target=publisher_runner.strategy.start, daemon=False)
    pub_thread.start()
    
    time.sleep(3)
    logger.info("Publisher running\n")
    
    # 2. Start Listener
    logger.info("Starting MockRedisListener...")
    listener_runner = IntegratedStrategyTestRunner(
        strategy_name="MockRedisListener",
        gui_type="RedisMessaging",
        duration_seconds=60,
        activate_trades=False,
        activate_bloomberg_mock=False,
        activate_redis=True,  # Listener needs redis
        market_update_interval=1.0
    )
    
    if not listener_runner.setup():
        logger.error("Listener setup failed")
        publisher_runner.teardown()
        exit(1)
    
    # Start listener
    for thread in listener_runner.threads:
        thread.start()
    
    listener_thread = threading.Thread(target=listener_runner.strategy.start, daemon=False)
    listener_thread.start()
    
    logger.info("Listener running\n")
    
    # 3. Run for 30 seconds
    logger.info("Running both strategies for 30s...")
    time.sleep(30)
    
    # 4. Cleanup
    logger.info("\nStopping...")
    listener_runner.teardown()
    publisher_runner.teardown()
    
    logger.info("Test complete")
