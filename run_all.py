"""
run_all.py
Run the complete Global Economic Digital Twin pipeline.
Executes all 5 modules in sequence.

Usage: python run_all.py
"""
import subprocess
import sys
import time

MODULES = [
    ("pipeline/data_pipeline.py",    "Module 1: Data Pipeline"),
    ("network/trade_network.py",     "Module 2: Trade Network"),
    ("simulation/shock_engine.py",   "Module 3: Shock Engine"),
    ("simulation/monte_carlo.py",    "Module 4: Monte Carlo"),
    ("ml/forecasting.py",            "Module 5: ML Forecasting"),
]

print("\n" + "█" * 65)
print("  GLOBAL ECONOMIC DIGITAL TWIN — Full Pipeline Run")
print("  yash752-stack")
print("█" * 65 + "\n")

start_total = time.time()
for script, name in MODULES:
    print(f"\n{'─'*65}")
    print(f"  Running {name}...")
    print(f"{'─'*65}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ {name} failed. Check errors above.")
        sys.exit(1)
    print(f"  ⏱  Completed in {elapsed:.1f}s")

total = time.time() - start_total
print(f"\n{'█'*65}")
print(f"  ✅ ALL MODULES COMPLETE in {total:.1f}s")
print(f"  Launch dashboard: streamlit run dashboard/app.py")
print(f"{'█'*65}\n")
