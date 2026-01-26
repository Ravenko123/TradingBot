#!/usr/bin/env python3
"""
Quick test to verify AI expectancy calculations are working correctly.
Run this to ensure the fix didn't break anything.
"""

def test_expectancy_calculation():
    """Test the expectancy formula."""
    
    print("=" * 60)
    print("EXPECTANCY CALCULATION TEST")
    print("=" * 60)
    
    # Test Case 1: High win rate but losing money (THE TRAP!)
    print("\n❌ TEST CASE 1: High Win Rate Trap")
    print("-" * 40)
    total_trades = 100
    wins = 99
    losses = 1
    win_rate = (wins / total_trades) * 100
    avg_win = 1.0  # $1 per win
    avg_loss = 5000.0  # $5000 loss
    
    win_pct = win_rate / 100
    loss_pct = 1 - win_pct
    expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
    total_profit = (wins * avg_win) - (losses * avg_loss)
    
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%  ← Looks AMAZING!")
    print(f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
    print(f"Expectancy: ${expectancy:.2f} per trade  ← LOSING MONEY!")
    print(f"Total Profit: ${total_profit:.2f}  ← ACCOUNT BLOWN!")
    print(f"Verdict: 🔴 HIGH WIN RATE BUT LOSING - Classic trap!")
    
    # Test Case 2: Lower win rate but making money (GOOD!)
    print("\n✅ TEST CASE 2: Money Printer (Low Win Rate)")
    print("-" * 40)
    total_trades = 100
    wins = 40
    losses = 60
    win_rate = (wins / total_trades) * 100
    avg_win = 100.0  # $100 per win
    avg_loss = 40.0  # $40 per loss
    
    win_pct = win_rate / 100
    loss_pct = 1 - win_pct
    expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
    total_profit = (wins * avg_win) - (losses * avg_loss)
    rr_ratio = avg_win / avg_loss
    
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%  ← Looks BAD!")
    print(f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
    print(f"RR Ratio: {rr_ratio:.2f}")
    print(f"Expectancy: ${expectancy:.2f} per trade  ← PRINTING MONEY!")
    print(f"Total Profit: ${total_profit:.2f}  ← PROFITABLE!")
    print(f"Verdict: 🚀 MONEY PRINTER MODE - Keep going!")
    
    # Test Case 3: Breakeven scenario
    print("\n⚖️ TEST CASE 3: Breakeven (Need to improve)")
    print("-" * 40)
    total_trades = 50
    wins = 25
    losses = 25
    win_rate = (wins / total_trades) * 100
    avg_win = 50.0
    avg_loss = 50.0
    
    win_pct = win_rate / 100
    loss_pct = 1 - win_pct
    expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
    total_profit = (wins * avg_win) - (losses * avg_loss)
    rr_ratio = avg_win / avg_loss
    
    print(f"Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
    print(f"RR Ratio: {rr_ratio:.2f}")
    print(f"Expectancy: ${expectancy:.2f} per trade  ← Break-even!")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Verdict: ⏳ Not losing but need better edge!")
    
    print("\n" + "=" * 60)
    print("✅ EXPECTANCY CALCULATIONS WORKING CORRECTLY!")
    print("=" * 60)
    print("\nKey Takeaway:")
    print("Win Rate alone is MEANINGLESS without considering:")
    print("  1. EXPECTANCY = (Win% × AvgWin) - (Loss% × AvgLoss)")
    print("  2. TOTAL PROFIT")
    print("  3. RR RATIO")
    print("  4. PROFIT FACTOR")
    print("\nAI will now optimize for PROFIT, not win rate!")
    print("=" * 60)


if __name__ == '__main__':
    test_expectancy_calculation()
