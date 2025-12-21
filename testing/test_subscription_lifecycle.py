"""
ESEMPIO COMPLETO DI TEST DEL NUOVO SISTEMA
"""

import time
import threading
import logging
from market_monitor_fi.live_data_hub.RTData import RTData
from market_monitor_fi.input_threads.EventHandler.BBGEventHandler import BBGEventHandler
from market_monitor_fi.input_threads.bloomberg.BloombergStreamingThread import BloombergStreamingThread

logging.basicConfig(level=logging.INFO)


def test_subscription_lifecycle():
    """Test complete subscription lifecycle"""

    # Setup
    lock = threading.Lock()
    rtdata = RTData(lock, fields=["BID", "ASK", "LAST"])

    print("=" * 60)
    print("TEST 1: Subscribe (should start as PENDING)")
    print("=" * 60)

    # Subscribe - starts as PENDING
    sub1 = rtdata.subscribe_bloomberg(
        id="AAPL",
        subscription_string="AAPL US Equity",
        fields=["BID", "ASK", "LAST"],
        params={"interval": 1}
    )

    sub2 = rtdata.subscribe_bloomberg(
        id="MSFT",
        subscription_string="MSFT US Equity",
        fields=["BID", "ASK", "LAST"],
        params={"interval": 1}
    )

    print(f"AAPL status: {sub1.status}")  # Should be "pending"
    print(f"MSFT status: {sub2.status}")  # Should be "pending"

    # Check pending
    pending = rtdata.get_pending_subscriptions("bloomberg")
    print(f"Pending subscriptions: {list(pending.keys())}")
    assert len(pending) == 2, "Should have 2 pending subscriptions"

    # Check health
    health = rtdata.get_subscription_health()
    print(f"Health: {health}")
    assert health['pending'] == 2, "Should have 2 pending"
    assert health['active'] == 0, "Should have 0 active"

    print("\n" + "=" * 60)
    print("TEST 2: Start Bloomberg Thread (will subscribe to pending)")
    print("=" * 60)

    # Start Bloomberg thread
    event_handler = BBGEventHandler(rtdata)
    bbg_thread = BloombergStreamingThread(event_handler)
    bbg_thread.start()

    # Wait for subscription to be processed
    print("Waiting for subscriptions to be processed...")
    time.sleep(10)

    # Check if activated
    pending = rtdata.get_pending_subscriptions("bloomberg")
    print(f"Pending subscriptions after 10s: {list(pending.keys())}")

    active = rtdata.get_all_subscriptions("bloomberg")
    print(f"Active subscriptions: {list(active.keys())}")

    health = rtdata.get_subscription_health()
    print(f"Health after activation: {health}")

    print("\n" + "=" * 60)
    print("TEST 3: Add new subscription while thread is running")
    print("=" * 60)

    # Add new subscription (should be picked up on next check)
    sub3 = rtdata.subscribe_bloomberg(
        id="GOOGL",
        subscription_string="GOOGL US Equity",
        fields=["BID", "ASK"],
        params={"interval": 1}
    )

    print(f"GOOGL status: {sub3.status}")  # "pending"

    # Wait for next check (5 seconds)
    print("Waiting for next periodic check (5 seconds)...")
    time.sleep(6)

    health = rtdata.get_subscription_health()
    print(f"Health after adding GOOGL: {health}")

    print("\n" + "=" * 60)
    print("TEST 4: Unsubscribe AAPL")
    print("=" * 60)

    # Unsubscribe
    rtdata.unsubscribe("AAPL", "bloomberg")
    print(f"AAPL status after unsubscribe: {sub1.status}")  # "closed"

    # Check unsubscribe queue
    to_unsub = rtdata.get_to_unsubscribe("bloomberg")
    print(f"To unsubscribe: {list(to_unsub.keys())}")
    assert "AAPL" in to_unsub, "AAPL should be in unsubscribe queue"

    # Wait for unsubscribe to be processed
    print("Waiting for unsubscribe to be processed...")
    time.sleep(6)

    to_unsub = rtdata.get_to_unsubscribe("bloomberg")
    print(f"To unsubscribe after processing: {list(to_unsub.keys())}")

    health = rtdata.get_subscription_health()
    print(f"Final health: {health}")

    print("\n" + "=" * 60)
    print("TEST 5: Monitor data")
    print("=" * 60)

    # Monitor data for 30 seconds
    for i in range(6):
        time.sleep(5)
        health = rtdata.get_subscription_health()
        data = rtdata.get_data_field()

        print(f"\n[{i * 5}s] Health: {health}")
        print(f"[{i * 5}s] Data shape: {data.shape if not data.empty else 'empty'}")
        print(f"[{i * 5}s] Securities: {list(data.index) if not data.empty else []}")

    # Stop
    print("\nStopping Bloomberg thread...")
    bbg_thread.stop()
    bbg_thread.join(timeout=5)

    print("\n✅ Test completed!")


def test_health_monitoring():
    """Test health monitoring"""

    lock = threading.Lock()
    rtdata = RTData(lock, fields=["BID", "ASK"])

    # Add subscriptions
    for i in range(10):
        rtdata.subscribe_bloomberg(
            id=f"TEST{i}",
            subscription_string=f"TEST{i} US Equity",
            fields=["BID", "ASK"]
        )

    # Check health
    health = rtdata.get_subscription_health()

    print("\nSubscription Health Report:")
    print(f"  Total: {health['total']}")
    print(f"  Pending: {health['pending']}")
    print(f"  Active: {health['active']}")
    print(f"  Failed: {health['failed']}")
    print(f"  To unsubscribe: {health['to_unsubscribe']}")

    print("\nBy Source:")
    for source, counts in health['by_source'].items():
        print(f"  {source}:")
        print(f"    Active: {counts['active']}")
        print(f"    Pending: {counts['pending']}")
        print(f"    To unsubscribe: {counts['to_unsubscribe']}")


if __name__ == "__main__":
    # Run tests
    print("Starting subscription lifecycle test...\n")
    test_subscription_lifecycle()

    print("\n" + "=" * 60)
    print("Starting health monitoring test...\n")
    test_health_monitoring()