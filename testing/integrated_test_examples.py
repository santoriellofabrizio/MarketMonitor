"""
Esempi di utilizzo del testing framework integrato.

Questo file mostra come:
1. Eseguire test di strategie singole
2. Eseguire test multipli sequenziali
3. Personalizzare i parametri
4. Analizzare i risultati
"""

import logging

from user_strategy.StrategyRegister import register_strategy
from testing.integrated_test_strategy_runner import run_integrated_test

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# ============================================================================
# ESEMPIO 1: Test rapido di SimplePriceMonitor
# ============================================================================

def example_simple_price_monitor_quick():
    """Test veloce di 10 secondi per verificare il flusso base."""
    print("\n" + "="*70)
    print("EXAMPLE 1: SimplePriceMonitor Quick Test (10 seconds)")
    print("="*70 + "\n")
    
    results = run_integrated_test(
        strategy_name="SimplePriceMonitor",
        duration_seconds=10,
    )
    
    if results['success']:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
        print("Errors:")
        for error in results['metrics']['errors']:
            print(f"  - {error}")


# ============================================================================
# EXEMPLE 2: Test TradeAccumulator con durata più lunga
# ============================================================================

def example_flow_detecting_strategy():
    """Test che accumula trades per 30 secondi."""
    print("\n" + "="*70)
    print("EXAMPLE 2: TradeAccumulator Test (30 seconds)")
    print("="*70 + "\n")

    from testing.test_strategy.TradeAccumulatorStrategy import TradeAccumulatorStrategy
    register_strategy("TradeAccumulator", TradeAccumulatorStrategy)
    
    results = run_integrated_test(
        strategy_name="FlowDetectingStrategy",
        duration_seconds=30,
        gui_type="RedisMessaging",
        trades_per_second=2.5,  # 2-3 trades/sec
    )
    
    if results['success']:
        print("✓ Test PASSED!")
        metrics = results['metrics']
        duration = metrics['duration']
        print(f"\nExecution time: {duration:.1f} seconds")
    else:
        print("✗ Test FAILED!")


# ============================================================================
# EXAMPLE 3: Test PriceSpreadAnalyzer con parametri personalizzati
# ============================================================================

def example_price_spread_analyzer_custom():
    """Test spread analyzer con solo 3 strumenti."""
    print("\n" + "="*70)
    print("EXAMPLE 3: PriceSpreadAnalyzer Custom Parameters")
    print("="*70 + "\n")
    
    results = run_integrated_test(
        strategy_name="PriceSpreadAnalyzer",
        duration_seconds=20,
        num_etf_instruments=3,  # Solo 3 ETF
        trades_per_second=2.5,
        market_update_interval=0.3,  # Aggiornamenti ogni 0.3s
    )
    
    if results['success']:
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")


# ============================================================================
# EXAMPLE 4: Test multipli sequenziali
# ============================================================================

def example_sequential_tests():
    """Esegue test di tutte e 3 le strategie."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Sequential Tests - All Strategies")
    print("="*70 + "\n")
    
    strategies = [
        ("SimplePriceMonitor", 10),
        ("TradeAccumulator", 15),
        ("PriceSpreadAnalyzer", 10),
    ]
    
    results_list = []
    
    for strategy_name, duration in strategies:
        print(f"\n>>> Testing {strategy_name} for {duration} seconds...")
        
        results = run_integrated_test(
            strategy_name=strategy_name,
            duration_seconds=duration,
        )
        
        status = "PASS ✓" if results['success'] else "FAIL ✗"
        results_list.append({
            "strategy": strategy_name,
            "duration": duration,
            "success": results['success'],
            "metrics": results['metrics'],
        })
        
        print(f"    Result: {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Sequential Tests")
    print("="*70)
    
    for result in results_list:
        status = "PASS" if result['success'] else "FAIL"
        duration = result['metrics']['duration']
        print(f"  {result['strategy']:30} {status:5} ({duration:.1f}s)")
    
    total_pass = sum(1 for r in results_list if r['success'])
    print(f"\nTotal: {total_pass}/{len(results_list)} passed")


# ============================================================================
# EXAMPLE 5: Test con alta frequenza (stress test)
# ============================================================================

def example_high_frequency_stress():
    """Stress test con alta frequenza di trades."""
    print("\n" + "="*70)
    print("EXAMPLE 5: High Frequency Stress Test")
    print("="*70 + "\n")
    
    results = run_integrated_test(
        strategy_name="TradeAccumulator",
        duration_seconds=30,
        trades_per_second=10.0,  # 10 trades/sec (alta frequenza)
        market_update_interval=0.1,  # Aggiornamenti frequenti
    )
    
    if results['success']:
        print("✓ Test PASSED!")
        print("High frequency test completed successfully")
    else:
        print("✗ Test FAILED!")


# ============================================================================
# EXAMPLE 6: Analisi dettagliata dei risultati
# ============================================================================

def example_detailed_analysis():
    """Esegue un test e mostra analisi dettagliata."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Detailed Analysis")
    print("="*70 + "\n")
    
    results = run_integrated_test(
        strategy_name="SimplePriceMonitor",
        duration_seconds=20,
    )
    
    if results['success']:
        print("✓ Test executed successfully\n")
        
        metrics = results['metrics']
        print("Metrics:")
        print(f"  - Duration: {metrics['duration']:.2f} seconds")
        print(f"  - Trades received: {metrics['trades_received']}")
        print(f"  - Market updates: {metrics['market_updates']}")
        
        if metrics['duration'] > 0:
            trades_per_sec = metrics['trades_received'] / metrics['duration']
            updates_per_sec = metrics['market_updates'] / metrics['duration']
            
            print(f"\nThroughput:")
            print(f"  - Trades/sec: {trades_per_sec:.2f}")
            print(f"  - Updates/sec: {updates_per_sec:.2f}")
        
        if metrics['errors']:
            print(f"\nErrors: {len(metrics['errors'])}")
            for error in metrics['errors']:
                print(f"  - {error}")
        else:
            print("\nNo errors!")
    else:
        print("✗ Test failed!")


# ============================================================================
# Main: esegui gli esempi
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║     MARKETMONITORFI INTEGRATION TESTING - EXAMPLES               ║")
    print("╚" + "="*68 + "╝")
    
    # Decomment la funzione che vuoi eseguire:
    
    # example_simple_price_monitor_quick()
    example_trade_accumulator()
    # example_price_spread_analyzer_custom()
    # example_sequential_tests()
    # example_high_frequency_stress()
    # example_detailed_analysis()
    
    print("\n" + "="*70)
    print("Examples completed")
    print("="*70)
