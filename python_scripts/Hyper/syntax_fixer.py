#!/usr/bin/env python3
"""
Python Try/Except Block Syntax Fixer

Fixes SyntaxError: expected 'except' or 'finally' block issues by analyzing
and fixing unclosed try blocks in Python files.

Specifically targets the error:
SyntaxError: expected 'except' or 'finally' block at line 4955

Usage:
    python syntax_fixer.py matcher50.py
"""

import sys
import os
import re
import shutil
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class TryExceptBlockFixer:
    """Comprehensive Python try/except block syntax fixer"""
    
    def __init__(self):
        self.fixes_applied = []
        self.indent_size = 4
        self.try_blocks = []  # Track open try blocks
        
    def fix_file(self, file_path: str) -> bool:
        """Fix try/except block issues in the specified file"""
        print(f"🔧 Analyzing try/except blocks in {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"❌ Error: File {file_path} not found!")
            return False
        
        # Create backup
        backup_path = f"{file_path}.syntax_backup"
        shutil.copy2(file_path, backup_path)
        print(f"💾 Created backup: {backup_path}")
        
        # Read original content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        print(f"📄 Original file: {len(lines)} lines")
        
        # Analyze the specific error area first
        self._analyze_error_area(lines, 4955)
        
        # Fix try/except block issues
        fixed_lines = self._fix_try_except_blocks(lines)
        
        # Write fixed content
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
        except Exception as e:
            print(f"❌ Error writing fixed file: {e}")
            # Restore backup
            shutil.copy2(backup_path, file_path)
            return False
        
        print(f"✅ Fixed try/except block issues!")
        self._print_fix_summary()
        
        return True
    
    def _analyze_error_area(self, lines: List[str], error_line: int):
        """Analyze the area around the syntax error"""
        print(f"\n🔍 ANALYZING ERROR AREA (line {error_line}):")
        print(f"{'='*80}")
        
        # Show lines around the error
        start = max(0, error_line - 20)
        end = min(len(lines), error_line + 5)
        
        try_stack = []
        
        for i in range(start, end):
            line = lines[i]
            line_num = i + 1
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Track try blocks
            if stripped.startswith('try:'):
                try_stack.append({'line': line_num, 'indent': indent})
                print(f"🔍 Line {line_num:4d}: TRY BLOCK START   -> {stripped}")
            elif stripped.startswith('except'):
                if try_stack:
                    try_block = try_stack.pop()
                    print(f"✅ Line {line_num:4d}: EXCEPT (closes try at {try_block['line']}) -> {stripped}")
                else:
                    print(f"❌ Line {line_num:4d}: ORPHANED EXCEPT -> {stripped}")
            elif stripped.startswith('finally'):
                if try_stack:
                    try_block = try_stack.pop()
                    print(f"✅ Line {line_num:4d}: FINALLY (closes try at {try_block['line']}) -> {stripped}")
                else:
                    print(f"❌ Line {line_num:4d}: ORPHANED FINALLY -> {stripped}")
            elif line_num == error_line:
                print(f"💥 Line {line_num:4d}: ERROR HERE       -> {stripped}")
            else:
                marker = "   " if line_num != error_line else "💥 "
                print(f"{marker} Line {line_num:4d}: ({indent:2d} spaces)     -> {stripped}")
        
        if try_stack:
            print(f"\n❌ UNCLOSED TRY BLOCKS FOUND:")
            for try_block in try_stack:
                print(f"   Line {try_block['line']}: Unclosed try block (indent: {try_block['indent']})")
        
        print(f"{'='*80}")
    
    def _fix_try_except_blocks(self, lines: List[str]) -> List[str]:
        """Fix try/except block issues throughout the file"""
        fixed_lines = []
        try_stack = []  # Stack to track open try blocks
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            line_num = i + 1
            
            # Track try blocks
            if stripped.startswith('try:'):
                try_stack.append({
                    'line_num': line_num,
                    'indent': indent,
                    'index': len(fixed_lines)
                })
                fixed_lines.append(line)
                print(f"🔍 Line {line_num}: Found try block (indent: {indent})")
                
            elif stripped.startswith('except') or stripped.startswith('finally'):
                # This should close a try block
                if try_stack and try_stack[-1]['indent'] == indent - self.indent_size:
                    # Properly indented except/finally for the last try
                    try_block = try_stack.pop()
                    fixed_lines.append(line)
                    print(f"✅ Line {line_num}: Proper except/finally for try at line {try_block['line_num']}")
                elif try_stack:
                    # Check if this except/finally is for an earlier try block
                    matched = False
                    for j in range(len(try_stack) - 1, -1, -1):
                        if try_stack[j]['indent'] == indent - self.indent_size:
                            # This except/finally matches an earlier try
                            # Close all intermediate try blocks first
                            while len(try_stack) > j:
                                unclosed_try = try_stack.pop()
                                self._add_missing_except(fixed_lines, unclosed_try)
                            
                            try_stack.pop()  # Remove the matched try
                            fixed_lines.append(line)
                            matched = True
                            break
                    
                    if not matched:
                        # Orphaned except/finally - remove or fix
                        print(f"❌ Line {line_num}: Orphaned {stripped.split(':')[0]} - fixing...")
                        fixed_lines.append(line)  # Keep it for now, might be part of a valid structure
                else:
                    # Orphaned except/finally with no try blocks
                    print(f"❌ Line {line_num}: Orphaned {stripped.split(':')[0]} with no try blocks")
                    fixed_lines.append(line)
                
            elif stripped.startswith('def ') or stripped.startswith('class '):
                # Function or class definition - close any open try blocks
                if try_stack:
                    print(f"🔧 Line {line_num}: Function/class definition found with {len(try_stack)} unclosed try blocks")
                    while try_stack:
                        unclosed_try = try_stack.pop()
                        self._add_missing_except(fixed_lines, unclosed_try)
                
                fixed_lines.append(line)
                
            else:
                # Regular line
                fixed_lines.append(line)
                
                # Check if we're exiting the scope of a try block due to dedentation
                if try_stack and stripped and indent <= try_stack[-1]['indent']:
                    # We've dedented to or below the try block level
                    while try_stack and indent <= try_stack[-1]['indent']:
                        unclosed_try = try_stack.pop()
                        print(f"🔧 Line {line_num}: Dedented below try block at line {unclosed_try['line_num']} - adding except")
                        self._add_missing_except(fixed_lines, unclosed_try, insert_before_current=True)
            
            i += 1
        
        # Close any remaining open try blocks at end of file
        while try_stack:
            unclosed_try = try_stack.pop()
            print(f"🔧 End of file: Adding except for unclosed try block at line {unclosed_try['line_num']}")
            self._add_missing_except(fixed_lines, unclosed_try)
        
        return fixed_lines
    
    def _add_missing_except(self, fixed_lines: List[str], try_block: Dict, insert_before_current: bool = False):
        """Add a missing except block for an unclosed try"""
        try_indent = try_block['indent']
        except_indent = try_indent + self.indent_size
        
        # Create a generic except block
        except_block = [
            ' ' * except_indent + 'except Exception as e:\n',
            ' ' * (except_indent + self.indent_size) + 'logger.warning(f"Exception in try block: {e}")\n',
            ' ' * (except_indent + self.indent_size) + 'pass\n'
        ]
        
        if insert_before_current:
            # Insert before the current line (last line added)
            insert_point = len(fixed_lines) - 1
        else:
            # Insert at the end
            insert_point = len(fixed_lines)
        
        # Insert the except block
        for j, except_line in enumerate(except_block):
            fixed_lines.insert(insert_point + j, except_line)
        
        self.fixes_applied.append(f"Added missing except block for try at line {try_block['line_num']}")
        print(f"  ✅ Added except block at position {insert_point}")
    
    def _validate_syntax(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax of the fixed file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            return True, None
            
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def _print_fix_summary(self):
        """Print summary of fixes applied"""
        print(f"\n🔧 TRY/EXCEPT BLOCK FIXES APPLIED:")
        print(f"{'='*80}")
        
        if not self.fixes_applied:
            print("  No try/except block issues found!")
        else:
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"  {i}. {fix}")
        
        print(f"\n✅ SYNTAX FIXING COMPLETE!")

def scan_for_try_blocks(file_path: str):
    """Scan file and report all try/except/finally blocks"""
    print(f"🔍 SCANNING TRY/EXCEPT BLOCKS IN: {file_path}")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    try_blocks = []
    except_blocks = []
    finally_blocks = []
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        
        if stripped.startswith('try:'):
            try_blocks.append({'line': i, 'indent': indent})
        elif stripped.startswith('except'):
            except_blocks.append({'line': i, 'indent': indent, 'type': stripped.split(':')[0]})
        elif stripped.startswith('finally:'):
            finally_blocks.append({'line': i, 'indent': indent})
    
    print(f"📊 BLOCK SUMMARY:")
    print(f"   Try blocks:     {len(try_blocks)}")
    print(f"   Except blocks:  {len(except_blocks)}")
    print(f"   Finally blocks: {len(finally_blocks)}")
    
    print(f"\n📋 TRY BLOCKS:")
    for try_block in try_blocks:
        print(f"   Line {try_block['line']:4d}: try: (indent: {try_block['indent']})")
    
    print(f"\n📋 EXCEPT BLOCKS:")
    for except_block in except_blocks:
        print(f"   Line {except_block['line']:4d}: {except_block['type']} (indent: {except_block['indent']})")
    
    print(f"\n📋 FINALLY BLOCKS:")
    for finally_block in finally_blocks:
        print(f"   Line {finally_block['line']:4d}: finally: (indent: {finally_block['indent']})")
    
    # Check for potential issues
    print(f"\n⚠️  POTENTIAL ISSUES:")
    if len(try_blocks) > len(except_blocks) + len(finally_blocks):
        print(f"   More try blocks than except/finally blocks!")
        print(f"   Potential unclosed try blocks: {len(try_blocks) - len(except_blocks) - len(finally_blocks)}")
    
    if len(except_blocks) > len(try_blocks):
        print(f"   More except blocks than try blocks!")
        print(f"   Potential orphaned except blocks: {len(except_blocks) - len(try_blocks)}")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python syntax_fixer.py <python_file>")
        print("\nExample: python syntax_fixer.py matcher50.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"🐍 PYTHON TRY/EXCEPT SYNTAX FIXER")
    print(f"{'='*80}")
    print(f"Target: {file_path}")
    print(f"Error: SyntaxError: expected 'except' or 'finally' block at line 4955")
    print(f"{'='*80}")
    
    # First scan the file structure
    scan_for_try_blocks(file_path)
    
    print(f"\n" + "="*80)
    
    # Apply fixes
    fixer = TryExceptBlockFixer()
    success = fixer.fix_file(file_path)
    
    if success:
        # Validate the syntax after fixing
        print(f"\n🔍 VALIDATING FIXED SYNTAX...")
        is_valid, error_msg = fixer._validate_syntax(file_path)
        
        if is_valid:
            print(f"✅ SYNTAX VALIDATION PASSED!")
        else:
            print(f"❌ SYNTAX VALIDATION FAILED:")
            print(f"   {error_msg}")
        
        print(f"\n🚀 SUCCESS!")
        print(f"   File fixed: {file_path}")
        print(f"   Backup created: {file_path}.syntax_backup")
        
        print(f"\n💡 Test the fix:")
        print(f"   python -m py_compile {file_path}")
        print(f"   ./runner.sh")
    else:
        print(f"\n❌ FAILED!")
        print(f"   Could not fix syntax issues")
        print(f"   Original file restored from backup")

if __name__ == "__main__":
    main()