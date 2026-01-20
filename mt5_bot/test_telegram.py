"""
Test script to verify Telegram bot configuration
Run this before using the EA to make sure notifications work
"""

import telegram_bot

def test_telegram():
    """Test if Telegram notifications are working"""
    
    print("="*60)
    print("TELEGRAM BOT TEST")
    print("="*60)
    print()
    
    # Test message
    test_message = "🧪 Test Message\n\nIf you see this, your Telegram bot is configured correctly!"
    
    print("Sending test message to Telegram...")
    print(f"Message: {test_message}")
    print()
    
    success = telegram_bot.send_telegram_message(test_message)
    
    print()
    print("="*60)
    if success:
        print("✅ SUCCESS! Check your Telegram to confirm.")
        print()
        print("Next steps:")
        print("1. Make sure you received the test message on Telegram")
        print("2. Open MetaTrader 5")
        print("3. Attach SimpleEA_EURUSD.ex5 to EURUSD M15 chart")
        print("4. Enable AutoTrading")
        print("5. Wait for trading signals!")
    else:
        print("❌ FAILED! Please check:")
        print("1. TELEGRAM_BOT_TOKEN is correct")
        print("2. TELEGRAM_CHAT_ID is correct")
        print("3. You have internet connection")
        print("4. You started a chat with your bot on Telegram")
    print("="*60)


if __name__ == "__main__":
    test_telegram()
