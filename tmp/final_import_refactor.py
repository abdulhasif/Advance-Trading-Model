import os
import re

def refactor_imports(root_dir):
    # Mapping for replacements
    replacements = [
        (re.compile(r'\bfrom core\.'), 'from trading_core.core.'),
        (re.compile(r'\bimport core\.'), 'import trading_core.core.'),
        # For the API, src represents its own internal logic
        # But for other projects importing API parts, it should be namespaced
        (re.compile(r'\bfrom src\.'), 'from trading_api.src.'),
        (re.compile(r'\bimport src\.'), 'import trading_api.src.'),
    ]
    
    for root, dirs, files in os.walk(root_dir):
        # Skip .git and __pycache__
        if '.git' in dirs: dirs.remove('.git')
        if '__pycache__' in dirs: dirs.remove('__pycache__')
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = content
                for pattern, replacement in replacements:
                    new_content = pattern.sub(replacement, new_content)
                
                if new_content != content:
                    print(f"Refactoring: {path}")
                    with open(path, 'w', encoding='utf-8', newline='') as f:
                        f.write(new_content)

if __name__ == "__main__":
    # Target modular directories
    dirs = ['trading_core', 'trading_pipeline', 'trading_engine', 'trading_api']
    for d in dirs:
        target = os.path.join(os.getcwd(), d)
        if os.path.exists(target):
            refactor_imports(target)
