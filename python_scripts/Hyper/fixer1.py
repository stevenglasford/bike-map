#!/usr/bin/env python3
"""
Targeted Python Syntax Fixer

Fixes specific syntax errors by analyzing the exact context and removing
incorrectly placed except blocks or fixing malformed try/except structures.

Usage:
    python targeted_syntax_fixer.py matcher50.py
"""

import sys
import os
import re
import shutil
import ast
from typing import List, Tuple, Dict

def analyze_line_context(lines: List[str], line_num: int, context_size: int = 10) -> Dict:
    """Analyze the context around a specific line"""
    line_index = line_num - 1  # Convert to 0-based index
    
    start = max(0, line_index - context_size)
    end = min(len(lines), line_index + context_size + 1)
    
    context = {
        'line_num': line_num,
        'line_content': lines[line_index].strip() if line_index < len(lines) else "",
        'context_lines': [],
        'problematic_patterns': []
    }
    
    for i in range(start, end):
        line = lines[i]
        line_number = i + 1
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        context['context_lines'].append({
            'line_num': line_number,
            'content': stripped,
            'indent': indent,
            'is_target': line_number == line_num
        })
        
        # Look for problematic patterns
        if line_number == line_num:
            if stripped.startswith('except') and i > 0:
                prev_line = lines[i-1].strip()
                if not prev_line.endswith(':') or prev_line.startswith('except') or prev_line.startswith('finally'):
                    context['problematic_patterns'].append("orphaned_except")
                    
        # Check for duplicate except blocks
        if stripped.startswith('except Exception as e:') and line_number != line_num:
            next_few_lines = []
            for j in range(i+1, min(i+4, len(lines))):
                next_few_lines.append(lines[j].strip())
            
            if any('logger.warning' in line and 'Unclosed try block' in line for line in next_few_lines):
                context['problematic_patterns'].append("auto_generated_except")
    
    return context

def fix_orphaned_except_blocks(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Remove orphaned except blocks that were incorrectly inserted"""
    fixed_lines = []
    removed_blocks = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for auto-generated except blocks that are causing problems
        if (stripped == 'except Exception as e:' and 
            i + 2 < len(lines) and
            'logger.warning' in lines[i+1] and
            'Unclosed try block' in lines[i+1] and
            lines[i+2].strip() == 'pass'):
            
            # This looks like an auto-generated except block
            # Check if it's properly positioned
            is_valid_except = False
            
            # Look backwards for a matching try block
            for j in range(i-1, max(0, i-10), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith('try:'):
                    # Found a try block, check if there's already an except
                    has_except = False
                    for k in range(j+1, i):
                        if lines[k].strip().startswith('except') or lines[k].strip().startswith('finally'):
                            has_except = True
                            break
                    
                    if not has_except:
                        is_valid_except = True
                    break
                elif prev_line.startswith('except') or prev_line.startswith('finally'):
                    # Found an existing except/finally, so this one is orphaned
                    break
            
            if not is_valid_except:
                # Remove the auto-generated except block
                removed_blocks.append(f"Lines {i+1}-{i+3}: Auto-generated except block")
                i += 3  # Skip the except, logger.warning, and pass lines
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return fixed_lines, removed_blocks

def fix_import_section_syntax(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Fix syntax issues specifically in the import section"""
    fixed_lines = []
    fixes_applied = []
    
    in_import_section = True
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check if we're still in the import section
        if stripped and not (stripped.startswith('import ') or 
                           stripped.startswith('from ') or
                           stripped.startswith('try:') or
                           stripped.startswith('except') or
                           stripped.startswith('finally') or
                           stripped.startswith('#') or
                           stripped.endswith('= True') or
                           stripped.endswith('= False') or
                           'AVAILABLE' in stripped):
            in_import_section = False
        
        # If we're in the import section and find problematic syntax
        if in_import_section:
            # Look for malformed try/except blocks in imports
            if stripped.startswith('except Exception as e:'):
                # Check if this except has a corresponding try
                has_matching_try = False
                for j in range(max(0, i-5), i):
                    if lines[j].strip().startswith('try:'):
                        # Check if there's already an except between try and this line
                        has_existing_except = False
                        for k in range(j+1, i):
                            if (lines[k].strip().startswith('except') and 
                                not lines[k].strip().startswith('except Exception as e:')):
                                has_existing_except = True
                                break
                        
                        if not has_existing_except:
                            has_matching_try = True
                        break
                
                if not has_matching_try:
                    # This is an orphaned except block in imports
                    fixes_applied.append(f"Removed orphaned except at line {i+1}")
                    # Skip this except block and its contents
                    while i < len(lines) and (lines[i].strip().startswith('except') or 
                                             'logger.warning' in lines[i] or
                                             lines[i].strip() == 'pass'):
                        i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    return fixed_lines, fixes_applied

def validate_and_show_syntax_error(file_path: str) -> Tuple[bool, Dict]:
    """Validate syntax and return detailed error info"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, {"valid": True, "message": "Syntax is valid"}
        
    except SyntaxError as e:
        return False, {
            "valid": False,
            "line": e.lineno,
            "message": e.msg,
            "text": e.text.strip() if e.text else "",
            "offset": e.offset
        }
    except Exception as e:
        return False, {
            "valid": False,
            "message": str(e)
        }

def fix_file_targeted(file_path: str) -> bool:
    """Apply targeted fixes based on the specific syntax error"""
    print(f"🎯 TARGETED SYNTAX FIXER: {file_path}")
    print("="*80)
    
    # Create backup
    backup_path = f"{file_path}.targeted_backup"
    shutil.copy2(file_path, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check current syntax error
    is_valid, error_info = validate_and_show_syntax_error(file_path)
    
    if is_valid:
        print("✅ File already has valid syntax!")
        return True
    
    print(f"❌ Syntax Error Details:")
    print(f"   Line: {error_info.get('line', 'unknown')}")
    print(f"   Message: {error_info.get('message', 'unknown')}")
    if 'text' in error_info:
        print(f"   Code: {error_info['text']}")
    
    # Analyze the error context
    if 'line' in error_info:
        context = analyze_line_context(lines, error_info['line'])
        
        print(f"\n🔍 CONTEXT ANALYSIS:")
        for ctx_line in context['context_lines']:
            marker = "💥" if ctx_line['is_target'] else "  "
            print(f"{marker} Line {ctx_line['line_num']:4d}: ({ctx_line['indent']:2d}) {ctx_line['content']}")
        
        if context['problematic_patterns']:
            print(f"\n❌ PROBLEMATIC PATTERNS DETECTED:")
            for pattern in context['problematic_patterns']:
                print(f"   • {pattern}")
    
    # Apply targeted fixes
    original_lines = lines[:]
    fixes_applied = []
    
    # Fix 1: Remove orphaned except blocks
    lines, removed_blocks = fix_orphaned_except_blocks(lines)
    fixes_applied.extend(removed_blocks)
    
    # Fix 2: Fix import section syntax issues
    lines, import_fixes = fix_import_section_syntax(lines)
    fixes_applied.extend(import_fixes)
    
    # Write fixed file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"\n🔧 FIXES APPLIED:")
        if fixes_applied:
            for fix in fixes_applied:
                print(f"   ✅ {fix}")
        else:
            print("   No automatic fixes could be applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        # Restore backup
        shutil.copy2(backup_path, file_path)
        return False

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python targeted_syntax_fixer.py <python_file>")
        print("\nExample: python targeted_syntax_fixer.py matcher50.py")
        print("\nThis fixer targets specific syntax errors and fixes them precisely.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        sys.exit(1)
    
    print("🎯 TARGETED PYTHON SYNTAX FIXER")
    print("="*80)
    print("This version analyzes the specific error and applies targeted fixes.")
    print("="*80)
    
    # Apply targeted fix
    success = fix_file_targeted(file_path)
    
    if success:
        # Validate after fix
        print(f"\n🔍 VALIDATING AFTER FIX:")
        is_valid, error_info = validate_and_show_syntax_error(file_path)
        
        if is_valid:
            print(f"   Status: ✅ VALID - Syntax errors fixed!")
            
            print(f"\n🚀 SUCCESS!")
            print(f"   ✅ File should now compile")
            print(f"   💾 Backup saved as: {file_path}.targeted_backup")
            
            print(f"\n💡 TEST THE FIX:")
            print(f"   python -m py_compile {file_path}")
            print(f"   ./runner.sh")
        else:
            print(f"   Status: ❌ STILL INVALID")
            print(f"   Line: {error_info.get('line', 'unknown')}")
            print(f"   Message: {error_info.get('message', 'unknown')}")
            
            if error_info.get('line'):
                print(f"\n🔍 REMAINING ERROR CONTEXT:")
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                context = analyze_line_context(lines, error_info['line'])
                for ctx_line in context['context_lines']:
                    marker = "💥" if ctx_line['is_target'] else "  "
                    print(f"{marker} Line {ctx_line['line_num']:4d}: {ctx_line['content']}")
            
            print(f"\n💡 MANUAL FIX NEEDED:")
            print(f"   The error may require manual correction")
            print(f"   Check the line shown above for syntax issues")
    else:
        print(f"\n❌ FAILED TO FIX")
        print(f"   Original file was restored from backup")

if __name__ == "__main__":
    main()