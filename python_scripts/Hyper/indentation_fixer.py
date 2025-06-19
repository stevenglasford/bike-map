#!/usr/bin/env python3
"""
Python Indentation Fixer

Fixes indentation errors in Python files, specifically targeting the error:
IndentationError: expected an indented block after function definition on line 479

Usage:
    python indentation_fixer.py matcher50.py
"""

import sys
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

class IndentationFixer:
    """Comprehensive Python indentation fixer"""
    
    def __init__(self):
        self.fixes_applied = []
        self.indent_size = 4  # Standard Python indentation
        
    def fix_file(self, file_path: str) -> bool:
        """Fix indentation issues in the specified file"""
        print(f"🔧 Analyzing indentation in {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"❌ Error: File {file_path} not found!")
            return False
        
        # Create backup
        backup_path = f"{file_path}.backup"
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
        
        # Analyze and fix indentation
        fixed_lines = self._fix_indentation(lines)
        
        # Write fixed content
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
        except Exception as e:
            print(f"❌ Error writing fixed file: {e}")
            # Restore backup
            shutil.copy2(backup_path, file_path)
            return False
        
        print(f"✅ Fixed indentation issues!")
        self._print_fix_summary()
        
        return True
    
    def _fix_indentation(self, lines: List[str]) -> List[str]:
        """Fix indentation issues in the lines"""
        fixed_lines = []
        
        # First pass: normalize tabs to spaces
        lines = self._normalize_tabs_to_spaces(lines)
        
        # Second pass: fix specific indentation issues
        lines = self._fix_function_indentation(lines)
        
        # Third pass: fix class indentation
        lines = self._fix_class_indentation(lines)
        
        # Fourth pass: fix general indentation consistency
        lines = self._fix_general_indentation(lines)
        
        # Fifth pass: fix specific line 481 area issues
        lines = self._fix_line_481_area(lines)
        
        return lines
    
    def _normalize_tabs_to_spaces(self, lines: List[str]) -> List[str]:
        """Convert all tabs to spaces"""
        fixed_lines = []
        tabs_found = 0
        
        for i, line in enumerate(lines, 1):
            if '\t' in line:
                tabs_found += 1
                # Convert tabs to spaces
                fixed_line = line.expandtabs(self.indent_size)
                fixed_lines.append(fixed_line)
                
                if tabs_found <= 10:  # Don't spam output
                    print(f"  Line {i}: Converted {line.count('t')} tabs to spaces")
            else:
                fixed_lines.append(line)
        
        if tabs_found > 0:
            self.fixes_applied.append(f"Converted {tabs_found} lines with tabs to spaces")
        
        return fixed_lines
    
    def _fix_function_indentation(self, lines: List[str]) -> List[str]:
        """Fix indentation issues after function definitions"""
        fixed_lines = []
        function_fixes = 0
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # Check for function definition
            if re.match(r'^\s*def\s+\w+\s*\(.*\).*:.*$', line.strip()):
                # This is a function definition
                current_indent = len(line) - len(line.lstrip())
                expected_body_indent = current_indent + self.indent_size
                
                # Check the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    
                    # Skip empty lines and comments
                    if not next_line.strip() or next_line.strip().startswith('#'):
                        continue
                    
                    # Skip docstrings
                    if '"""' in next_line or "'''" in next_line:
                        continue
                    
                    # Check indentation of first non-empty line
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    if next_indent <= current_indent and next_line.strip():
                        # This line should be indented as function body
                        spaces_needed = expected_body_indent - next_indent
                        if spaces_needed > 0:
                            fixed_next_line = ' ' * spaces_needed + next_line.lstrip()
                            lines[j] = fixed_next_line
                            function_fixes += 1
                            print(f"  Line {j+1}: Fixed function body indentation (+{spaces_needed} spaces)")
                    
                    break  # Only fix the first line of function body
        
        if function_fixes > 0:
            self.fixes_applied.append(f"Fixed {function_fixes} function body indentation issues")
        
        return lines
    
    def _fix_class_indentation(self, lines: List[str]) -> List[str]:
        """Fix indentation issues after class definitions"""
        fixed_lines = []
        class_fixes = 0
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # Check for class definition
            if re.match(r'^\s*class\s+\w+.*:.*$', line.strip()):
                # This is a class definition
                current_indent = len(line) - len(line.lstrip())
                expected_body_indent = current_indent + self.indent_size
                
                # Check the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    
                    # Skip empty lines and comments
                    if not next_line.strip() or next_line.strip().startswith('#'):
                        continue
                    
                    # Skip docstrings
                    if '"""' in next_line or "'''" in next_line:
                        continue
                    
                    # Check indentation of first non-empty line
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    if next_indent <= current_indent and next_line.strip():
                        # This line should be indented as class body
                        spaces_needed = expected_body_indent - next_indent
                        if spaces_needed > 0:
                            fixed_next_line = ' ' * spaces_needed + next_line.lstrip()
                            lines[j] = fixed_next_line
                            class_fixes += 1
                            print(f"  Line {j+1}: Fixed class body indentation (+{spaces_needed} spaces)")
                    
                    break  # Only fix the first line of class body
        
        if class_fixes > 0:
            self.fixes_applied.append(f"Fixed {class_fixes} class body indentation issues")
        
        return lines
    
    def _fix_general_indentation(self, lines: List[str]) -> List[str]:
        """Fix general indentation consistency issues"""
        fixed_lines = []
        general_fixes = 0
        
        for i, line in enumerate(lines):
            if not line.strip():  # Empty line
                fixed_lines.append(line)
                continue
            
            # Calculate current indentation
            indent = len(line) - len(line.lstrip())
            
            # Check if indentation is a multiple of indent_size
            if indent % self.indent_size != 0 and indent > 0:
                # Round to nearest proper indentation
                proper_indent = round(indent / self.indent_size) * self.indent_size
                fixed_line = ' ' * proper_indent + line.lstrip()
                fixed_lines.append(fixed_line)
                general_fixes += 1
                print(f"  Line {i+1}: Normalized indentation ({indent} -> {proper_indent} spaces)")
            else:
                fixed_lines.append(line)
        
        if general_fixes > 0:
            self.fixes_applied.append(f"Fixed {general_fixes} general indentation issues")
        
        return fixed_lines
    
    def _fix_line_481_area(self, lines: List[str]) -> List[str]:
        """Specifically fix the area around line 481 where the error occurs"""
        if len(lines) < 481:
            return lines
        
        area_fixes = 0
        
        # Look at lines around 481 (479-485)
        start_line = max(0, 478)  # Line 479 in 0-based indexing
        end_line = min(len(lines), 485)
        
        print(f"🔍 Analyzing critical area (lines {start_line+1}-{end_line})...")
        
        for i in range(start_line, end_line):
            if i >= len(lines):
                break
                
            line = lines[i]
            line_num = i + 1
            
            print(f"  Line {line_num}: '{line.rstrip()}'")
            
            # Check for the specific pattern that causes the error
            if 'device = video_features_batch.device' in line:
                # This line should be indented as part of a function/method
                current_indent = len(line) - len(line.lstrip())
                
                # Look for the function definition above this line
                for j in range(i - 1, max(0, i - 20), -1):
                    prev_line = lines[j]
                    if re.match(r'^\s*def\s+\w+.*:.*$', prev_line):
                        # Found function definition
                        func_indent = len(prev_line) - len(prev_line.lstrip())
                        expected_indent = func_indent + self.indent_size
                        
                        if current_indent != expected_indent:
                            # Fix the indentation
                            fixed_line = ' ' * expected_indent + line.lstrip()
                            lines[i] = fixed_line
                            area_fixes += 1
                            print(f"  ✅ Line {line_num}: Fixed critical indentation error")
                            print(f"     Before: {current_indent} spaces")
                            print(f"     After:  {expected_indent} spaces")
                        break
            
            # Check for missing indentation after colons
            if i > 0 and ':' in lines[i-1] and not lines[i-1].strip().endswith('"""') and not lines[i-1].strip().endswith("'''"):
                prev_line = lines[i-1]
                # If previous line ends with colon and current line is not indented properly
                if (prev_line.strip().endswith(':') and 
                    line.strip() and 
                    not line.startswith(' ' * self.indent_size) and
                    not line.strip().startswith('#')):
                    
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    expected_indent = prev_indent + self.indent_size
                    current_indent = len(line) - len(line.lstrip())
                    
                    if current_indent < expected_indent:
                        fixed_line = ' ' * expected_indent + line.lstrip()
                        lines[i] = fixed_line
                        area_fixes += 1
                        print(f"  ✅ Line {line_num}: Fixed indentation after colon")
        
        if area_fixes > 0:
            self.fixes_applied.append(f"Fixed {area_fixes} critical indentation issues around line 481")
        
        return lines
    
    def _print_fix_summary(self):
        """Print summary of fixes applied"""
        print(f"\n🔧 INDENTATION FIXES APPLIED:")
        print(f"{'='*60}")
        
        if not self.fixes_applied:
            print("  No indentation issues found!")
        else:
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"  {i}. {fix}")
        
        print(f"\n✅ INDENTATION FIXING COMPLETE!")
        print(f"   The script should now run without IndentationError")

def analyze_file_structure(file_path: str):
    """Analyze file structure to help identify issues"""
    print(f"📊 ANALYZING FILE STRUCTURE: {file_path}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Look at the specific error area
    if len(lines) >= 481:
        print(f"🔍 Error area analysis (lines 475-485):")
        for i in range(474, min(485, len(lines))):
            line = lines[i]
            line_num = i + 1
            indent = len(line) - len(line.lstrip())
            
            marker = "👉 " if line_num == 481 else "   "
            print(f"{marker}Line {line_num:3d} ({indent:2d} spaces): {line.rstrip()}")
    
    # Check for mixed indentation
    tab_lines = sum(1 for line in lines if '\t' in line)
    space_lines = sum(1 for line in lines if line.startswith(' ') and not line.startswith('\t'))
    
    print(f"\n📏 Indentation analysis:")
    print(f"   Lines with tabs: {tab_lines}")
    print(f"   Lines with spaces: {space_lines}")
    print(f"   Total lines: {len(lines)}")
    
    if tab_lines > 0 and space_lines > 0:
        print(f"   ⚠️ Mixed tabs and spaces detected!")
    
    # Look for function definitions around the error
    print(f"\n🔍 Function definitions around line 481:")
    for i, line in enumerate(lines):
        if re.match(r'^\s*def\s+\w+.*:.*$', line.strip()):
            line_num = i + 1
            if 470 <= line_num <= 490:
                indent = len(line) - len(line.lstrip())
                print(f"   Line {line_num}: {line.strip()} (indent: {indent})")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python indentation_fixer.py <python_file>")
        print("\nExample: python indentation_fixer.py matcher50.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"🐍 PYTHON INDENTATION FIXER")
    print(f"{'='*60}")
    print(f"Target: {file_path}")
    print(f"Error: IndentationError at line 481")
    print(f"{'='*60}")
    
    # First analyze the file structure
    analyze_file_structure(file_path)
    
    print(f"\n" + "="*60)
    
    # Apply fixes
    fixer = IndentationFixer()
    success = fixer.fix_file(file_path)
    
    if success:
        print(f"\n🚀 SUCCESS!")
        print(f"   File fixed: {file_path}")
        print(f"   Backup created: {file_path}.backup")
        print(f"   Try running your script again!")
        
        print(f"\n💡 Test the fix:")
        print(f"   python -m py_compile {file_path}")
        print(f"   ./runner.sh")
    else:
        print(f"\n❌ FAILED!")
        print(f"   Could not fix indentation issues")
        print(f"   Original file restored from backup")

if __name__ == "__main__":
    main()