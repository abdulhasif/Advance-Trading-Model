import os
import subprocess
import re

def get_git_content(path):
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], encoding="utf-8")

def extract_comp(content, name, type="def"):
    lines = content.split('\n')
    start = -1
    for i, line in enumerate(lines):
        if line.startswith(f"{type} {name}"):
            start = i
            break
    
    if start == -1: return ""
    
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() == "": continue
        if not line.startswith(" "):
            if line.startswith("def ") or line.startswith("class ") or line.startswith("# =="):
                 end = i
                 break
    return "\n".join(lines[start:end]).strip()

# Target file mappings
file_map = {
    "velocity.py": [("compute_velocity", "def"), ("compute_velocity_long", "def")],
    "momentum_acceleration.py": [("compute_momentum_acceleration", "def")],
    "volatility_context.py": [("compute_tib_zscore", "def"), ("compute_vpb_roc", "def"), ("compute_squeeze_zscore", "def")],
    "vwap_zscore.py": [("compute_vwap_zscore", "def")],
    "vpt_acceleration.py": [("compute_vpt_acceleration", "def")],
    "wick_pressure.py": [("compute_wick_pressure", "def")],
    "streak_exhaustion.py": [("compute_streak_exhaustion", "def"), ("compute_consecutive_same_dir", "def")],
    "structural_trend.py": [("compute_structural_score", "def")],
    "order_flow.py": [("compute_order_flow_delta", "def")],
    "relative_strength.py": [("RelativeStrengthCalculator", "class")],
    "market_regime.py": [("compute_market_regime_dummies", "def")],
}

try:
    features_content = get_git_content("src/core/features.py")
    base_path = r"c:\Trading Platform\Advance Trading Model\trading_core\core\features\calculators"
    
    for file_name, components in file_map.items():
        logic_blocks = []
        for name, comp_type in components:
            block = extract_comp(features_content, name, comp_type)
            if block:
                logic_blocks.append(block)
            else:
                print(f"CRITICAL: Could not find {comp_type} {name}")
                
        if logic_blocks:
            content = f"import numpy as np\nimport pandas as pd\nfrom core.config import base_config as config\n\n"
            content += "\n\n".join(logic_blocks) + "\n"
            full_path = os.path.join(base_path, file_name)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"SUCCESS: Written {file_name}")

except Exception as e:
    print(f"ERROR: {e}")
