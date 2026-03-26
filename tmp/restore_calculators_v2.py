import os
import subprocess
import re

def get_git_content(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], encoding="utf-8")

def extract_func(content, name):
    # Regex to find top-level function definition
    pattern = rf"^def {name}\(.*\):"
    lines = content.split('\n')
    start = -1
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start = i
            break
    
    if start == -1: return ""
    
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() == "": continue
        # Next top-level def or class
        if re.match(r"^(def |class |# =+)", line):
             end = i
             break
    return "\n".join(lines[start:end]).strip()

calculators = {
    "compute_velocity": "velocity.py",
    "compute_momentum_acceleration": "momentum_acceleration.py",
    "compute_vwap_zscore": "vwap_zscore.py",
    "compute_vpt_acceleration": "vpt_acceleration.py",
    "compute_wick_pressure": "wick_pressure.py",
    "compute_squeeze_zscore": "squeeze_zscore.py",
    "compute_streak_exhaustion": "streak_exhaustion.py",
    "compute_consecutive_same_dir": "consecutive_same_dir.py",
    "compute_structural_score": "structural_trend.py"
}

try:
    features_content = get_git_content("src/core/features.py")
    base_path = r"c:\Trading Platform\Advance Trading Model\trading_core\core\features\calculators"
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for func_name, file_name in calculators.items():
        logic = extract_func(features_content, func_name)
        if not logic:
            print(f"CRITICAL: Could not find {func_name}")
            continue
            
        content = f"import numpy as np\nimport pandas as pd\nfrom core.config import base_config as config\n\n{logic}\n"
        full_path = os.path.join(base_path, file_name)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"SUCCESS: Written {len(content)} bytes to {file_name}")

except Exception as e:
    print(f"ERROR: {e}")
