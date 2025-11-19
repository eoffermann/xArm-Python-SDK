"""
reset_ethernet.py

This script disables and re-enables a specified Ethernet interface on Windows,
useful for resolving connectivity issues with the robotic arms.
It checks for administrative privileges and self-elevates if necessary.

Platform: Windows
Python version: 3.10.11 (Anaconda)
Dependencies: Standard Library Only
"""

import ctypes
import subprocess
import sys
import os
import time
import argparse

# ================================
# Configuration
# ================================
ETHERNET_INTERFACE_NAME = "Ethernet"  # Change if your interface has a different name
DISABLE_WAIT_SECONDS = 3              # Time to wait after disabling before re-enabling

# ================================
# Utility Functions
# ================================
def is_admin():
    """
    Check if the script is running with administrative privileges.

    Returns:
        bool: True if admin, False otherwise.
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def run_as_admin():
    """Relaunch the script with administrative privileges using a temporary VBS script."""
    script_path = os.path.abspath(sys.argv[0])
    params = ' '.join([f'"{arg}"' for arg in sys.argv[1:]])
    vbs_content = f'''
Set UAC = CreateObject("Shell.Application")
UAC.ShellExecute "pythonw.exe", "\"{script_path}\" {params}", "", "runas", 1
'''
    vbs_path = os.path.join(os.environ['TEMP'], 'elevate.vbs')
    with open(vbs_path, 'w') as vbs_file:
        vbs_file.write(vbs_content)
    subprocess.call(['cscript', '//nologo', vbs_path])
    sys.exit(0)


def toggle_ethernet(interface_name: str, wait_seconds: int = 3):
    """
    Disable and re-enable the specified Ethernet interface using netsh.

    Args:
        interface_name (str): Name of the interface to toggle.
        wait_seconds (int): Seconds to wait between disable and enable.

    Raises:
        subprocess.CalledProcessError: If netsh commands fail.
    """
    print(f"Disabling interface: {interface_name}...")
    subprocess.check_call([
        "netsh", "interface", "set", "interface",
        interface_name, "admin=disabled"
    ])
    print(f"Waiting {wait_seconds} seconds...")
    time.sleep(wait_seconds)
    print(f"Enabling interface: {interface_name}...")
    subprocess.check_call([
        "netsh", "interface", "set", "interface",
        interface_name, "admin=enabled"
    ])
    print("Interface reset completed.")


# ================================
# Main Function
# ================================
def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Reset a Windows Ethernet interface by disabling and re-enabling it."
    )
    parser.add_argument(
        "--interface", "-i",
        default=ETHERNET_INTERFACE_NAME,
        help=f"Name of the Ethernet interface to reset (default: {ETHERNET_INTERFACE_NAME})"
    )
    parser.add_argument(
        "--wait", "-w",
        type=int,
        default=DISABLE_WAIT_SECONDS,
        help="Seconds to wait after disabling before enabling (default: 3)"
    )
    args = parser.parse_args()

    if not is_admin():
        print("Script not running as administrator. Attempting to elevate...")
        run_as_admin()

    try:
        toggle_ethernet(args.interface, args.wait)
    except subprocess.CalledProcessError as e:
        print(f"Error toggling interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
