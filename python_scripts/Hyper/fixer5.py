#!/usr/bin/env python3
"""
Comprehensive auto-fix script for matcher50.py
More thorough approach with validation and error checking.
"""

import os
import re
import shutil
import ast
import sys
from datetime import datetime
from pathlib import Path

def create_backup(file_path):
    """Create a backup of the original file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return backup_path

def validate_python_syntax(content):
    """Check if the Python code has valid syntax"""
    try:
        ast.parse(content)
        return True, "Syntax is valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def fix_main_function_comprehensive(content):
    """Comprehensive fix for main function - handles multiple patterns"""
    
    # Pattern 1: Look for the end of config setup
    patterns_to_try = [
        # Original pattern
        r'(\s+logger\.info\(f"💾 RAM Cache: \{\'✅\' if \'ram_cache_manager\' in locals\(\) and ram_cache_manager else \'❌\'\} \(\{config\.ram_cache_gb:.1f\}GB\)"\)\s*)',
        # Alternative pattern - look for RAM cache info line
        r'(\s+logger\.info\(f".*RAM Cache.*"\)\s*)',
        # Fallback - look for the end of feature status logging
        r'(\s+logger\.info\(f".*Enhanced GPS Processing.*"\)\s*)',
        # Last resort - look for powersafe status
        r'(\s+logger\.info\(f".*PowerSafe Mode.*"\)\s*)'
    ]
    
    processing_code = '''
        # ========== CALL THE ACTUAL PROCESSING SYSTEM ==========
        try:
            logger.info("🚀 Starting complete turbo processing system...")
            results = complete_turbo_video_gpx_correlation_system(args, config)
            
            if results:
                logger.info(f"✅ Processing completed successfully with {len(results)} results")
                print(f"\\n🎉 SUCCESS: Processing completed with {len(results)} video results!")
                return 0
            else:
                logger.error("❌ Processing completed but returned no results")
                print(f"\\n⚠️ Processing completed but no results were generated")
                return 1
                
        except KeyboardInterrupt:
            logger.info("🛑 Processing interrupted by user")
            print(f"\\n⚠️ Processing interrupted by user")
            return 130
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            if args.debug:
                import traceback
                logger.error(f"Full traceback:\\n{traceback.format_exc()}")
            print(f"\\n❌ PROCESSING FAILED: {e}")
            print(f"\\n🔧 Try running with --debug for more detailed error information")
            return 1

'''
    
    for i, pattern in enumerate(patterns_to_try):
        if re.search(pattern, content):
            replacement = f'\\g<1>{processing_code}'
            content = re.sub(pattern, replacement, content)
            print(f"✅ Fixed: Added main function processing call (pattern {i+1})")
            return content
    
    # If no patterns match, try to add before the except blocks
    except_pattern = r'(\s+)(except KeyboardInterrupt:)'
    if re.search(except_pattern, content):
        replacement = f'\\g<1>{processing_code}\\n\\g<1>\\g<2>'
        content = re.sub(except_pattern, replacement, content)
        print("✅ Fixed: Added main function processing call (before except blocks)")
        return content
    
    print("⚠️  Warning: Could not find insertion point for main function completion")
    return content

def fix_all_device_issues(content):
    """Comprehensive device issue fixes"""
    fixes_applied = []
    
    # Fix 1: DistortionAwareAttention device parameter
    if "device=device" in content and "nn.Parameter(torch.ones(1, 1, 8, 16, device=device))" in content:
        content = content.replace(
            "nn.Parameter(torch.ones(1, 1, 8, 16, device=device))",
            "nn.Parameter(torch.ones(1, 1, 8, 16))"
        )
        fixes_applied.append("DistortionAwareAttention device parameter")
    
    # Fix 2: Add device variable to _extract_complete_features
    pattern = r'(def _extract_complete_features\(self, video_path: str, gpu_id: int\) -> Optional\[Dict\]:\s+"""[^"]*"""\s+try:\s+)(# Load and preprocess video)'
    replacement = r'\1device = torch.device(f"cuda:{gpu_id}")  # FIXED: Define device variable\n            \n            \2'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        fixes_applied.append("_extract_complete_features device variable")
    
    # Fix 3: Add device variable to _extract_visual_features
    pattern = r'(def _extract_visual_features\(self, frames_tensor: torch\.Tensor, gpu_id: int\) -> Dict\[str, np\.ndarray\]:\s+"""[^"]*"""\s+try:\s+)(batch_size, num_frames, channels, height, width = frames_tensor\.shape)'
    replacement = r'\1device = frames_tensor.device  # FIXED: Get device from tensor\n            \2'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        fixes_applied.append("_extract_visual_features device variable")
    
    # Fix 4: DistortionAwareAttention forward method
    if "def forward(self, features):" in content and "# Apply channel attention" in content:
        pattern = r'(def forward\(self, features\):\s+)(# Apply channel attention)'
        replacement = r'\1# Get device from input features\n                device = features.device\n                \n                \2'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            fixes_applied.append("DistortionAwareAttention forward method device")
    
    # Fix 5: Distortion weights device movement
    if "self.distortion_weights" in content and "F.interpolate" in content:
        pattern = r'(# Resize distortion weights to match feature map\s+)(dist_weights = F\.interpolate\(\s+self\.distortion_weights,)'
        replacement = r'\1# Move distortion weights to correct device\n                dist_weights = self.distortion_weights.to(device)\n                \n                # Resize distortion weights to match feature map\n                dist_weights = F.interpolate(\n                    dist_weights,'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            fixes_applied.append("distortion weights device movement")
    
    for fix in fixes_applied:
        print(f"✅ Fixed: {fix}")
    
    return content

def fix_ram_cache_issues(content):
    """Fix all RAM cache related issues"""
    fixes_applied = []
    
    # Fix class name mismatch
    if "RAMCacheManager(config)" in content:
        content = content.replace(
            "RAMCacheManager(config)",
            "TurboRAMCacheManager(config, config.ram_cache_gb)"
        )
        fixes_applied.append("RAM cache manager class name")
    
    # Fix any other RAM cache references
    if "ram_cache_manager = RAMCacheManager" in content:
        content = content.replace(
            "ram_cache_manager = RAMCacheManager",
            "ram_cache_manager = TurboRAMCacheManager"
        )
        fixes_applied.append("RAM cache manager assignment")
    
    for fix in fixes_applied:
        print(f"✅ Fixed: {fix}")
    
    return content

def fix_exception_handling(content):
    """Fix problematic exception handling"""
    fixes_applied = []
    
    # Remove problematic "unclosed try block" exceptions
    if "Unclosed try block exception" in content:
        content = content.replace(
            'logger.warning(f"Unclosed try block exception: {e}")',
            'logger.warning(f"Exception: {e}")'
        )
        fixes_applied.append("exception logging messages")
    
    # Fix empty except blocks
    pattern = r'except Exception as e:\s+logger\.warning\(f"[^"]*"\)\s+pass\s+except Exception:'
    if re.search(pattern, content):
        content = re.sub(pattern, 'except Exception:', content)
        fixes_applied.append("duplicate exception blocks")
    
    for fix in fixes_applied:
        print(f"✅ Fixed: {fix}")
    
    return content

def fix_main_call(content):
    """Fix the main function call at the end"""
    # Multiple patterns to try
    patterns = [
        (r'if __name__ == "__main__":\s+main\(\)', 'if __name__ == "__main__":\n    exit_code = main()\n    sys.exit(exit_code)'),
        (r'if __name__ == "__main__":\s+sys\.exit\(main\(\)\)', 'if __name__ == "__main__":\n    exit_code = main()\n    sys.exit(exit_code)'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            print("✅ Fixed: Main function call with proper exit code")
            return content
    
    # If no pattern matches, try to add it at the end
    if 'if __name__ == "__main__"' not in content:
        content += '\n\nif __name__ == "__main__":\n    exit_code = main()\n    sys.exit(exit_code)\n'
        print("✅ Fixed: Added main function call at end of file")
    
    return content

def add_missing_imports(content):
    """Add any missing imports that might be needed"""
    fixes_applied = []
    
    # Check for sys import (needed for sys.exit)
    if 'sys.exit' in content and 'import sys' not in content:
        # Find import section and add sys
        import_pattern = r'(import os\nimport glob)'
        if re.search(import_pattern, content):
            content = re.sub(import_pattern, r'\1\nimport sys', content)
            fixes_applied.append("sys import")
    
    for fix in fixes_applied:
        print(f"✅ Fixed: Added missing {fix}")
    
    return content

def comprehensive_syntax_check(content):
    """More thorough syntax checking and fixing"""
    issues_found = []
    
    # Check for common issues
    if content.count('"""') % 2 != 0:
        issues_found.append("Unmatched triple quotes")
    
    if content.count("'''") % 2 != 0:
        issues_found.append("Unmatched triple quotes (single)")
    
    # Check for unmatched parentheses in common patterns
    for line_num, line in enumerate(content.split('\n'), 1):
        if line.count('(') != line.count(')'):
            if any(keyword in line for keyword in ['logger.info', 'print', 'if ', 'for ', 'while ']):
                issues_found.append(f"Unmatched parentheses on line {line_num}")
                break
    
    if issues_found:
        print("⚠️  Potential syntax issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("   You may need to fix these manually.")
    
    return content

def main():
    """Main function with comprehensive fixing"""
    
    # Find matcher50.py file
    current_dir = Path.cwd()
    matcher_file = current_dir / "matcher50.py"
    
    if not matcher_file.exists():
        print("❌ Error: matcher50.py not found in current directory")
        print(f"Current directory: {current_dir}")
        print("Please run this script from the directory containing matcher50.py")
        return 1
    
    print(f"🔧 Found matcher50.py: {matcher_file}")
    print(f"📁 Working directory: {current_dir}")
    print()
    
    # Create backup
    backup_path = create_backup(matcher_file)
    print()
    
    # Read the file
    try:
        with open(matcher_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Read file: {len(content)} characters, {len(content.split())} lines")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 1
    
    # Initial syntax check
    is_valid, msg = validate_python_syntax(content)
    print(f"📝 Initial syntax check: {'✅' if is_valid else '❌'} {msg}")
    print()
    
    print("🔧 Applying comprehensive fixes...")
    print("=" * 60)
    
    # Apply all fixes in order
    original_content = content
    
    content = fix_main_function_comprehensive(content)
    content = fix_ram_cache_issues(content)
    content = fix_all_device_issues(content)
    content = fix_exception_handling(content)
    content = fix_main_call(content)
    content = add_missing_imports(content)
    content = comprehensive_syntax_check(content)
    
    print("=" * 60)
    
    # Final syntax check
    is_valid_after, msg_after = validate_python_syntax(content)
    print(f"📝 Final syntax check: {'✅' if is_valid_after else '❌'} {msg_after}")
    
    # Check if any changes were made
    if content == original_content:
        print("⚠️  No changes were made. The file may already be fixed or have different structure.")
        print("This could mean:")
        print("  • The file is already properly fixed")
        print("  • The code structure differs from expected patterns")
        print("  • Some fixes may need to be applied manually")
        return 0
    
    # Show change summary
    original_lines = len(original_content.split('\n'))
    new_lines = len(content.split('\n'))
    print(f"📊 Changes: {original_lines} → {new_lines} lines ({new_lines - original_lines:+} lines)")
    
    # Write the fixed file only if syntax is valid
    if not is_valid_after:
        print("❌ Cannot save file: Syntax errors detected")
        print("The original file has been preserved.")
        return 1
    
    try:
        with open(matcher_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed file written: {matcher_file}")
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        print(f"Restoring from backup: {backup_path}")
        shutil.copy2(backup_path, matcher_file)
        return 1
    
    print(f"\n🎉 COMPREHENSIVE FIXES COMPLETED!")
    print(f"📁 Backup saved as: {backup_path}")
    print(f"📝 Fixed file: {matcher_file}")
    print()
    print(f"🚀 Test your script:")
    print(f"   python matcher50.py -d /path/to/your/data --debug")
    print()
    print(f"💡 If you still get errors:")
    print(f"   1. Check the debug output for specific error messages")
    print(f"   2. Verify your data directory contains video and GPX files")
    print(f"   3. Make sure all dependencies are installed")
    print(f"   4. Try: python -c 'import torch; print(torch.cuda.is_available())'")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)