import sys
from pathlib import Path
import re
import os

# Add the project root to sys.path to find 'config'
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

all_usage = set()
src_dir = ROOT / 'src'

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith('.py'):
            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(r'config\.([A-Z_0-9]+)', content)
                all_usage.update(matches)

# Check root files
for file in ['main.py', 'offline_spoofer.py', 'replay_inspector.py']:
    path = ROOT / file
    if path.exists():
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            matches = re.findall(r'config\.([A-Z_0-9]+)', content)
            all_usage.update(matches)

missing = [attr for attr in sorted(list(all_usage)) if not hasattr(config, attr)]
print('\n'.join(missing))
