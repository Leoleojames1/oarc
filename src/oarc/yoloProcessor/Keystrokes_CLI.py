#!/usr/bin/env python3

"""
🐭⌨️ Macro CLI

This script provides a command-line interface (CLI) for creating, listing, and executing macros
for mouse and keyboard inputs.

Key Features:
1. Create macros with mouse and keyboard actions
2. List existing macros
3. Execute saved macros
4. Save macros to a file for persistence

Usage example:
    python macro_cli.py create "my_macro" --actions "mouse_move 100 100" "key_press a" "mouse_click"
    python macro_cli.py list
    python macro_cli.py execute "my_macro"

Make sure to install the required library:
    pip install pynput
    
    #TODO THIS NEEDS TO BE INTEGRATED AS A CLASS FOR THE YOLO VISION AGENTS SO THAT AGENTS CAN ACCESS 
    # TOOLING TO CREATE MACROS AND CONTROL SCREEN MOVEMENTS BASED ON YOLO VISION
@LBorcherding
"""

import argparse
import json
import logging
import time
from pynput import mouse, keyboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MACRO_FILE = "macros.json"

def load_macros():
    try:
        with open(MACRO_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_macros(macros):
    with open(MACRO_FILE, "w") as f:
        json.dump(macros, f, indent=2)

def run_create(args):
    """
    Create a new macro with the specified actions.
    
    Args:
        args: Parsed command-line arguments
    """
    macros = load_macros()
    macros[args.name] = args.actions
    save_macros(macros)
    logger.info(f"Created macro '{args.name}' with {len(args.actions)} actions")

def run_list(args):
    """
    List all saved macros.
    
    Args:
        args: Parsed command-line arguments
    """
    macros = load_macros()
    if not macros:
        logger.info("No macros found")
    else:
        for name, actions in macros.items():
            logger.info(f"Macro: {name}")
            for action in actions:
                logger.info(f"  - {action}")

def run_execute(args):
    """
    Execute the specified macro.
    
    Args:
        args: Parsed command-line arguments
    """
    macros = load_macros()
    if args.name not in macros:
        logger.error(f"Macro '{args.name}' not found")
        return

    mouse_controller = mouse.Controller()
    keyboard_controller = keyboard.Controller()

    for action in macros[args.name]:
        parts = action.split()
        if parts[0] == "mouse_move":
            mouse_controller.move(int(parts[1]), int(parts[2]))
        elif parts[0] == "mouse_click":
            mouse_controller.click(mouse.Button.left)
        elif parts[0] == "key_press":
            keyboard_controller.press(parts[1])
            keyboard_controller.release(parts[1])
        time.sleep(0.1)  # Small delay between actions

    logger.info(f"Executed macro '{args.name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🐭⌨️ Macro CLI for Mouse and Keyboard Inputs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command parser
    create_parser = subparsers.add_parser("create", help="Create a new macro")
    create_parser.add_argument("name", type=str, help="Name of the macro")
    create_parser.add_argument("--actions", nargs="+", required=True, help="List of actions for the macro")

    # List command parser
    list_parser = subparsers.add_parser("list", help="List all macros")

    # Execute command parser
    execute_parser = subparsers.add_parser("execute", help="Execute a macro")
    execute_parser.add_argument("name", type=str, help="Name of the macro to execute")

    args = parser.parse_args()

    if args.command == "create":
        run_create(args)
    elif args.command == "list":
        run_list(args)
    elif args.command == "execute":
        run_execute(args)
    else:
        parser.print_help()