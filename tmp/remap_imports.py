import os

mappings = {
    'from src.core': 'from core',
    'from src.live': 'from trading_engine.src',
    'from src.ml': 'from trading_pipeline.scripts',
    'from src.data': 'from trading_pipeline.pipeline'
}

projects = [
    r'c:\Trading Platform\Advance Trading Model\trading_core',
    r'c:\Trading Platform\Advance Trading Model\trading_pipeline',
    r'c:\Trading Platform\Advance Trading Model\trading_engine',
    r'c:\Trading Platform\Advance Trading Model\trading_api'
]

for project_root in projects:
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for target, replacement in mappings.items():
                        content = content.replace(target, replacement)
                    
                    if content != original_content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Updated imports in {path}")
                except Exception as e:
                    print(f"Error in {path}: {e}")
