"""
cli_utils.py
Shared ANSI terminal helpers used across all coverage-app modules.
Works on Windows Terminal / PowerShell >= 5 (ANSI escape codes supported).
"""

def _a(code):  return f"\033[{code}m"

RST    = _a(0)
BOLD   = _a(1)
DIM    = _a(2)
CYAN   = _a(36)
GREEN  = _a(32)
YELLOW = _a(33)
RED    = _a(31)

def _banner():
	w = 56
	bar = "─" * w
	print(f"\n{CYAN}{BOLD}┌{bar}┐{RST}")
	print(f"{CYAN}{BOLD}│{'CBRS  Coverage Analysis Tool':^{w}}│{RST}")
	print(f"{CYAN}{BOLD}└{bar}┘{RST}\n")

def _section(title):
	print(f"\n{BOLD}{CYAN}── {title} {'─' * max(1, 48 - len(title))}{RST}")

def _info(msg):   print(f"  {CYAN}ℹ{RST}  {msg}")
def _ok(msg):     print(f"  {GREEN}✔{RST}  {msg}")
def _warn(msg):   print(f"  {YELLOW}⚠{RST}  {msg}")
def _err(msg):    print(f"  {RED}✖{RST}  {msg}")
def _wait(msg):   input(f"\n  {YELLOW}▶{RST}  {msg}")

def _ask(prompt, default=None):
	hint = f" {DIM}[{default}]{RST}" if default is not None else ""
	return input(f"  {YELLOW}›{RST}  {prompt}{hint}: ").strip()

def _ask_int(prompt, default):
	while True:
		raw = _ask(prompt, default=default)
		if raw == "":
			return default
		try:
			return int(raw)
		except ValueError:
			_warn("Please enter a valid integer value.")

def _ask_float(prompt, default):
	while True:
		raw = _ask(prompt, default=default)
		if raw == "":
			return default
		try:
			return float(raw)
		except ValueError:
			_warn("Please enter a valid numeric value.")
