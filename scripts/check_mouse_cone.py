"""
anti_sleep.py

Keeps your computer session active by slightly moving the mouse periodically.
Works on Windows, macOS, and Linux.

Requirements:
    pip install pyautogui
"""

import pyautogui  # library to control mouse/keyboard
import time       # for sleep intervals

# Safety: move mouse to top-left corner to stop script instantly
pyautogui.FAILSAFE = True

def keep_awake(interval=60):
    """
    Moves the mouse slightly every `interval` seconds.

    Args:
        interval (int): Time in seconds between movements
    """
    print("Anti-sleep script running... Press Ctrl+C to stop.")

    try:
        while True:
            # Get current mouse position
            x, y = pyautogui.position()

            # Move mouse slightly and back (invisible to user)
            pyautogui.moveTo(x + 1, y + 1, duration=0.1)
            pyautogui.moveTo(x, y, duration=0.1)

            # Wait before next action
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    keep_awake(interval=5)  # change interval if needed