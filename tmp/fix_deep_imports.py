import os

deep_mappings = {
    'from core.renko': 'from core.physics.renko',
    'from core.quant_fixes': 'from core.physics.quant_fixes',
    'from core.hybrid_news': 'from core.physics.hybrid_news',
    'from core.risk': 'from core.risk.risk_fortress',
    'from core.strategy': 'from core.risk.strategy',
    'from core.execution_guard': 'from core.risk.execution_guard',
    # And potentially feature imports in core.features module
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
                    for target, replacement in deep_mappings.items():
                        content = content.replace(target, replacement)
                    
                    if content != original_content:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Updated deep imports in {path}")
                except Exception as e:
                    print(f"Error in {path}: {e}")
