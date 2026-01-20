"""
Telegram Notification Bot for MT5 Trading EA
Educational Graduation Project 2026

This simple script sends trading notifications to your Telegram chat.
"""

import sys
import requests
from datetime import datetime

# ============================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"      # Get from @userinfobot on Telegram

# ============================================
# SCRIPT LOGIC (No need to modify below)
# ============================================

def send_telegram_message(message):
    """
    Send a message to Telegram using the Bot API
    
    Args:
        message (str): Message text to send
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Validate configuration
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ERROR: Please configure your TELEGRAM_BOT_TOKEN in telegram_bot.py")
        return False
    
    if TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("❌ ERROR: Please configure your TELEGRAM_CHAT_ID in telegram_bot.py")
        return False
    
    # Build API URL
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Prepare message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"🤖 MT5 Trading Bot\n{timestamp}\n\n{message}"
    
    # Prepare request payload
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": full_message,
        "parse_mode": "HTML"
    }
    
    try:
        # Send POST request to Telegram API
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Telegram message sent successfully!")
            return True
        else:
            print(f"❌ Telegram API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False


def main():
    """Main entry point when script is called from MQL5"""
    
    # Check if message was provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python telegram_bot.py \"Your message here\"")
        print("\nExample:")
        print('python telegram_bot.py "🟢 BUY Trade Opened | Symbol: EURUSD | Price: 1.08500"')
        return
    
    # Get message from command line arguments
    message = sys.argv[1]
    
    # Send the message
    success = send_telegram_message(message)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
