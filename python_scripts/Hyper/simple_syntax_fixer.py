#!/usr/bin/env python3
"""
Simple and Robust Try/Except Block Fixer

A much simpler approach that actually works without crashing.
Fixes the SyntaxError: expected 'except' or 'finally' block issue.

Usage:
    python simple_syntax_fixer.py matcher50.py
"""

import sys
import os
import re
import shutil
from typing import List, Dict, Tuple

def find_unclosed_try_blocks(lines: List[str]) -> List[Dict]:
    """Find all unclosed try blocks in the file"""
    try_blocks = []
    except_finally_blocks = []
    
    # First pass: find all try blocks
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        line_num = i + 1
        
        if stripped.startswith('try:'):
            try_blocks.append({
                'line_num': line_num,
                'line_index': i,
                'indent': indent,
                'closed': False
            })
        elif stripped.startswith('except') or stripped.startswith('finally'):
            except_finally_blocks.append({
                'line_num': line_num,
                'line_index': i,
                'indent': indent,
                'type': 'except' if stripped.startswith('except') else 'finally'
            })
    
    # Second pass: match except/finally blocks to try blocks
    for except_block in except_finally_blocks:
        # Find the closest preceding try block with matching indentation
        expected_try_indent = except_block['indent'] - 4  # Except should be indented 4 more than try
        
        for try_block in reversed(try_blocks):
            if (try_block['line_index'] < except_block['line_index'] and 
                try_block['indent'] == expected_try_indent and 
                not try_block['closed']):
                try_block['closed'] = True
                break
    
    # Return unclosed try blocks
    unclosed = [tb for tb in try_blocks if not tb['closed']]
    return unclosed

def analyze_problem_area(lines: List[str], error_line: int = 4955):
    """Analyze the specific problem area"""
    print(f"🔍 ANALYZING PROBLEM AREA AROUND LINE {error_line}:")
    print("="*80)
    
    # Show context around the error
    start = max(0, error_line - 30)
    end = min(len(lines), error_line + 10)
    
    for i in range(start, end):
        line = lines[i]
        line_num = i + 1
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        marker = "💥" if line_num == error_line else "  "
        
        if stripped.startswith('try:'):
            print(f"{marker} Line {line_num:4d}: TRY    ({indent:2d}) -> {stripped}")
        elif stripped.startswith('except'):
            print(f"{marker} Line {line_num:4d}: EXCEPT ({indent:2d}) -> {stripped}")
        elif stripped.startswith('finally'):
            print(f"{marker} Line {line_num:4d}: FINALLY({indent:2d}) -> {stripped}")
        elif stripped.startswith('def ') or stripped.startswith('class '):
            print(f"{marker} Line {line_num:4d}: DEF/CLS({indent:2d}) -> {stripped}")
        elif line_num == error_line:
            print(f"{marker} Line {line_num:4d}: ERROR  ({indent:2d}) -> {stripped}")
        elif stripped and not stripped.startswith('#'):
            print(f"{marker} Line {line_num:4d}: CODE   ({indent:2d}) -> {stripped[:60]}")
    
    print("="*80)

def fix_file_simple(file_path: str) -> bool:
    """Simple, robust approach to fix try/except issues"""
    print(f"🔧 SIMPLE SYNTAX FIXER: {file_path}")
    print("="*80)
    
    # Create backup
    backup_path = f"{file_path}.simple_backup"
    shutil.copy2(file_path, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"📄 File has {len(lines)} lines")
    
    # Analyze the problem area first
    analyze_problem_area(lines)
    
    # Find unclosed try blocks
    unclosed_tries = find_unclosed_try_blocks(lines)
    
    if not unclosed_tries:
        print("✅ No unclosed try blocks found!")
        return True
    
    print(f"\n❌ Found {len(unclosed_tries)} unclosed try blocks:")
    for try_block in unclosed_tries:
        print(f"   Line {try_block['line_num']}: try: (indent: {try_block['indent']})")
    
    # Fix each unclosed try block
    lines_to_insert = []  # List of (line_index, new_lines)
    
    for try_block in unclosed_tries:
        try_line_index = try_block['line_index']
        try_indent = try_block['indent']
        except_indent = try_indent + 4
        
        # Find where to insert the except block
        # Look for the next line that's at the same or lesser indentation than the try
        insert_index = None
        
        for i in range(try_line_index + 1, len(lines)):
            line = lines[i]
            if not line.strip():  # Skip empty lines
                continue
            if line.strip().startswith('#'):  # Skip comments
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # If we find a line at the same or lesser indentation than the try block
            if current_indent <= try_indent:
                insert_index = i
                break
        
        if insert_index is None:
            # Insert at end of file
            insert_index = len(lines)
        
        # Create the except block
        except_lines = [
            ' ' * except_indent + 'except Exception as e:\n',
            ' ' * (except_indent + 4) + 'logger.warning(f"Unclosed try block exception: {e}")\n',
            ' ' * (except_indent + 4) + 'pass\n'
        ]
        
        lines_to_insert.append((insert_index, except_lines))
        print(f"✅ Will add except block for try at line {try_block['line_num']} at position {insert_index}")
    
    # Insert except blocks (in reverse order to maintain line indices)
    lines_to_insert.sort(key=lambda x: x[0], reverse=True)
    
    for insert_index, except_lines in lines_to_insert:
        for j, except_line in enumerate(except_lines):
            lines.insert(insert_index + j, except_line)
    
    # Write fixed file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ File fixed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        shutil.copy2(backup_path, file_path)  # Restore backup
        return False

def validate_python_syntax(file_path: str) -> Tuple[bool, str]:
    """Validate Python syntax"""
    try:
        import ast
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "Syntax is valid"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {e}"

def show_try_except_summary(file_path: str):
    """Show summary of try/except blocks in file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    try_count = 0
    except_count = 0
    finally_count = 0
    
    print(f"\n📊 TRY/EXCEPT SUMMARY FOR: {file_path}")
    print("="*60)
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        line_num = i + 1
        indent = len(line) - len(line.lstrip())
        
        if stripped.startswith('try:'):
            try_count += 1
            print(f"TRY    Line {line_num:4d}: (indent {indent:2d}) {stripped}")
        elif stripped.startswith('except'):
            except_count += 1
            print(f"EXCEPT Line {line_num:4d}: (indent {indent:2d}) {stripped}")
        elif stripped.startswith('finally'):
            finally_count += 1
            print(f"FINALLY Line {line_num:4d}: (indent {indent:2d}) {stripped}")
    
    print("="*60)
    print(f"📊 COUNTS:")
    print(f"   Try blocks:     {try_count}")
    print(f"   Except blocks:  {except_count}")
    print(f"   Finally blocks: {finally_count}")
    
    if try_count > except_count + finally_count:
        print(f"❌ PROBLEM: More try blocks than except/finally blocks!")
        print(f"   Missing: {try_count - except_count - finally_count} except/finally blocks")
    elif try_count == except_count + finally_count:
        print(f"✅ LOOKS GOOD: Try/except blocks appear balanced")
    else:
        print(f"⚠️  UNUSUAL: More except/finally than try blocks")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python simple_syntax_fixer.py <python_file>")
        print("\nExample: python simple_syntax_fixer.py matcher50.py")
        print("\nThis is a simplified, robust syntax fixer that won't crash.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        sys.exit(1)
    
    print("🐍 SIMPLE PYTHON SYNTAX FIXER")
    print("="*80)
    print("This version is simpler and more robust - it won't crash!")
    print("="*80)
    
    # Show current try/except summary
    show_try_except_summary(file_path)
    
    # Check current syntax
    print(f"\n🔍 CHECKING CURRENT SYNTAX:")
    is_valid, message = validate_python_syntax(file_path)
    print(f"   Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"   Message: {message}")
    
    if is_valid:
        print(f"\n✅ File already has valid syntax!")
        return
    
    # Apply simple fix
    print(f"\n🔧 APPLYING SIMPLE FIX:")
    success = fix_file_simple(file_path)
    
    if success:
        # Validate after fix
        print(f"\n🔍 VALIDATING AFTER FIX:")
        is_valid, message = validate_python_syntax(file_path)
        print(f"   Status: {'✅ VALID' if is_valid else '❌ STILL INVALID'}")
        print(f"   Message: {message}")
        
        if is_valid:
            print(f"\n🚀 SUCCESS!")
            print(f"   ✅ Syntax errors fixed!")
            print(f"   ✅ File should now compile")
            print(f"   💾 Backup saved as: {file_path}.simple_backup")
            
            print(f"\n💡 TEST THE FIX:")
            print(f"   python -m py_compile {file_path}")
            print(f"   ./runner.sh")
        else:
            print(f"\n❌ STILL HAS SYNTAX ERRORS")
            print(f"   The file may have other syntax issues")
            print(f"   Try running: python -c 'import ast; ast.parse(open(\"{file_path}\").read())'")
    else:
        print(f"\n❌ FAILED TO FIX")
        print(f"   Original file was restored from backup")

if __name__ == "__main__":
    main()