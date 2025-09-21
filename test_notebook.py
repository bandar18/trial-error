#!/usr/bin/env python3
"""
Test script to validate the Upstox Trade Monitor notebook
"""

import json
import sys

def test_notebook_structure():
    """Test the notebook structure and content"""
    try:
        # Load notebook
        with open('upstox_trade_monitor.ipynb', 'r') as f:
            notebook = json.load(f)
        
        print("🧪 Testing Upstox Trade Monitor Notebook...")
        print("=" * 50)
        
        # Test 1: Basic structure
        assert 'cells' in notebook, "Notebook missing cells"
        assert 'metadata' in notebook, "Notebook missing metadata"
        
        cells = notebook['cells']
        print(f"✅ Notebook has {len(cells)} cells")
        
        # Test 2: Check for required sections
        required_sections = [
            "CONFIG - API Credentials",
            "Imports and Dependencies", 
            "Helpers - Authentication and API Client",
            "Data Pipelines",
            "Calculations",
            "IV-RV Context Dashboard",
            "Trade-Position Monitor",
            "Self-Tests and Validation"
        ]
        
        markdown_cells = [cell for cell in cells if cell['cell_type'] == 'markdown']
        section_titles = []
        for cell in markdown_cells:
            source = ''.join(cell['source'])
            if source.startswith('## '):
                section_titles.append(source.strip())
        
        print(f"✅ Found {len(section_titles)} markdown sections")
        
        # Test 3: Check for code cells
        code_cells = [cell for cell in cells if cell['cell_type'] == 'code']
        print(f"✅ Found {len(code_cells)} code cells")
        
        # Test 4: Check for key classes
        all_code = '\n'.join([''.join(cell['source']) for cell in code_cells])
        
        key_classes = [
            'UpstoxAPIClient',
            'UniverseBuilder', 
            'VolatilityAnalytics',
            'PortfolioAnalytics',
            'IVRVDashboard',
            'PositionMonitor',
            'MainDashboard',
            'DashboardTests'
        ]
        
        for class_name in key_classes:
            if f'class {class_name}' in all_code:
                print(f"✅ Found class: {class_name}")
            else:
                print(f"❌ Missing class: {class_name}")
        
        # Test 5: Check for key functions
        key_functions = [
            'calculate_iv_rank',
            'calculate_realized_volatility',
            'calculate_atr',
            'calculate_expected_move',
            'is_trading_hours'
        ]
        
        for func_name in key_functions:
            if f'def {func_name}' in all_code:
                print(f"✅ Found function: {func_name}")
            else:
                print(f"❌ Missing function: {func_name}")
        
        # Test 6: Check for imports
        required_imports = [
            'pandas', 'numpy', 'httpx', 'plotly', 'ipywidgets', 'loguru'
        ]
        
        for import_name in required_imports:
            if f'import {import_name}' in all_code or f'from {import_name}' in all_code:
                print(f"✅ Found import: {import_name}")
            else:
                print(f"❌ Missing import: {import_name}")
        
        print("=" * 50)
        print("🎉 Notebook structure validation completed!")
        print("\n📋 Summary:")
        print(f"  - Total cells: {len(cells)}")
        print(f"  - Markdown cells: {len(markdown_cells)}")
        print(f"  - Code cells: {len(code_cells)}")
        print(f"  - Key classes: {sum(1 for c in key_classes if f'class {c}' in all_code)}/{len(key_classes)}")
        print(f"  - Key functions: {sum(1 for f in key_functions if f'def {f}' in all_code)}/{len(key_functions)}")
        print(f"  - Required imports: {sum(1 for i in required_imports if f'import {i}' in all_code or f'from {i}' in all_code)}/{len(required_imports)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing notebook: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_notebook_structure()
    sys.exit(0 if success else 1)