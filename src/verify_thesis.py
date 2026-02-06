# ==============================================================================
# FINAL VERIFICATION SCRIPT
# ==============================================================================
# Run this BEFORE thesis submission to check everything is correct
# ==============================================================================

import os
import json
import pickle
from datetime import datetime

THESIS_DIR = '/content/drive/MyDrive/thesis'

print("=" * 70)
print("FINAL THESIS VERIFICATION")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

errors = []
warnings = []

# ==============================================================================
# CHECK 1: FOLDER STRUCTURE
# ==============================================================================
print("\n📁 CHECK 1: Folder Structure")
print("-" * 50)

required_folders = ['notebooks', 'results', 'figures', 'src', 'docs', 'data', 'thesis_writing']
for folder in required_folders:
    path = f'{THESIS_DIR}/{folder}'
    if os.path.exists(path):
        count = len(os.listdir(path))
        print(f"   ✅ {folder}/ ({count} files)")
    else:
        print(f"   ❌ {folder}/ MISSING")
        errors.append(f"Missing folder: {folder}")

# Check for duplicate folders (should NOT exist)
bad_folders = ['experiments', 'plots']
for folder in bad_folders:
    path = f'{THESIS_DIR}/{folder}'
    if os.path.exists(path):
        print(f"   ⚠️  {folder}/ EXISTS (should be removed)")
        warnings.append(f"Duplicate folder exists: {folder}")

# ==============================================================================
# CHECK 2: NOTEBOOKS
# ==============================================================================
print("\n📓 CHECK 2: Notebooks")
print("-" * 50)

notebooks = [
    '01_setup_test.ipynb',
    '02_ioi_reproduction.ipynb',
    '03_nl_explanation_generator.ipynb',
    '04_baselines_comparison.ipynb',
    '05_expanded_evaluation.ipynb',
    '06_failure_analysis_main.ipynb',
    '07_esnli_format_study.ipynb',
    '08_learned_nl_generator.ipynb',
    '09_template_vs_learned.ipynb',
    '10_final_evaluation.ipynb',
]

for nb in notebooks:
    path = f'{THESIS_DIR}/notebooks/{nb}'
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
        
        has_seed = 'manual_seed' in content or 'np.random.seed' in content
        has_paths = 'RESULTS_DIR' in content
        
        if has_seed and has_paths:
            print(f"   ✅ {nb}")
        elif has_seed:
            print(f"   ⚠️  {nb} - missing RESULTS_DIR path")
            warnings.append(f"{nb}: missing paths")
        else:
            print(f"   ❌ {nb} - missing seeds")
            errors.append(f"{nb}: missing reproducibility seeds")
    else:
        print(f"   ❌ {nb} - FILE NOT FOUND")
        errors.append(f"{nb}: file not found")

# ==============================================================================
# CHECK 3: RESULTS FILES
# ==============================================================================
print("\n📊 CHECK 3: Results Files")
print("-" * 50)

required_results = [
    '02_ioi_reproduction',
    '03_nl_explanation',
    '04_baselines',
    '05_expanded_evaluation',
    '06_failure_analysis',
    '07_esnli_study',
    '08_learned_nl',
    '09_template_vs_learned',
    '10_final_summary',
]

results_dir = f'{THESIS_DIR}/results'
if os.path.exists(results_dir):
    result_files = os.listdir(results_dir)
    for req in required_results:
        found = any(req in f for f in result_files)
        if found:
            matching = [f for f in result_files if req in f][0]
            print(f"   ✅ {matching}")
        else:
            print(f"   ❌ {req}*.pkl NOT FOUND")
            errors.append(f"Missing result: {req}")
else:
    print("   ❌ results/ folder not found")
    errors.append("results/ folder missing")

# ==============================================================================
# CHECK 4: FIGURES
# ==============================================================================
print("\n🖼️  CHECK 4: Figures")
print("-" * 50)

required_figures = [
    'fig_06',
    'fig_07',
    'fig_08',
    'fig_09',
    'fig_10',
    'ioi_circuit',
    'baseline_comparison',
]

figures_dir = f'{THESIS_DIR}/figures'
if os.path.exists(figures_dir):
    figure_files = os.listdir(figures_dir)
    for req in required_figures:
        found = any(req in f for f in figure_files)
        if found:
            print(f"   ✅ {req}*.png found")
        else:
            print(f"   ⚠️  {req}*.png not found")
            warnings.append(f"Missing figure: {req}")
else:
    print("   ❌ figures/ folder not found")
    errors.append("figures/ folder missing")

# ==============================================================================
# CHECK 5: SOURCE FILES
# ==============================================================================
print("\n💻 CHECK 5: Source Files")
print("-" * 50)

required_src = ['demo_pipeline.py', '00_reproducibility_config.py']
src_dir = f'{THESIS_DIR}/src'

for req in required_src:
    path = f'{src_dir}/{req}'
    if os.path.exists(path):
        print(f"   ✅ {req}")
    else:
        print(f"   ❌ {req} NOT FOUND")
        errors.append(f"Missing source: {req}")

# ==============================================================================
# CHECK 6: DOCUMENTATION
# ==============================================================================
print("\n📄 CHECK 6: Documentation")
print("-" * 50)

required_docs = ['README.md', 'requirements.txt']
for doc in required_docs:
    path = f'{THESIS_DIR}/{doc}'
    if os.path.exists(path):
        print(f"   ✅ {doc}")
    else:
        print(f"   ❌ {doc} NOT FOUND")
        errors.append(f"Missing: {doc}")

docs_files = ['CITATIONS.md', 'LIMITATIONS.md', 'RELATED_WORK.md']
for doc in docs_files:
    path = f'{THESIS_DIR}/docs/{doc}'
    if os.path.exists(path):
        print(f"   ✅ docs/{doc}")
    else:
        print(f"   ⚠️  docs/{doc} not found")
        warnings.append(f"Missing doc: {doc}")

# ==============================================================================
# CHECK 7: CANONICAL RESULTS CONSISTENCY
# ==============================================================================
print("\n🔬 CHECK 7: Canonical Results")
print("-" * 50)

canonical = {
    'sufficiency': 1.00,
    'comprehensiveness': 0.22,
    'f1': 0.36,
    'circuit_coverage': 0.614,
    'template_quality': 0.60,
    'learned_quality': 0.98,
}

# Try to load results and check consistency
try:
    results_files = os.listdir(results_dir)
    summary_file = [f for f in results_files if '10_final' in f]
    if summary_file:
        with open(f'{results_dir}/{summary_file[0]}', 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ Loaded {summary_file[0]}")
        
        # Check if canonical values match (approximate)
        if 'canonical' in data:
            for key, expected in canonical.items():
                if key in data['canonical']:
                    actual = data['canonical'][key]
                    if abs(actual - expected) < 0.05:
                        print(f"   ✅ {key}: {actual:.2f} (expected {expected:.2f})")
                    else:
                        print(f"   ⚠️  {key}: {actual:.2f} (expected {expected:.2f})")
                        warnings.append(f"Result mismatch: {key}")
except Exception as e:
    print(f"   ⚠️  Could not verify results: {e}")
    warnings.append("Could not verify canonical results")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n❌ ERRORS: {len(errors)}")
for err in errors:
    print(f"   • {err}")

print(f"\n⚠️  WARNINGS: {len(warnings)}")
for warn in warnings:
    print(f"   • {warn}")

print("\n" + "-" * 70)
if len(errors) == 0:
    print("🎉 VERIFICATION PASSED - Ready for submission!")
else:
    print(f"❌ VERIFICATION FAILED - Fix {len(errors)} error(s) before submission")
print("-" * 70)
