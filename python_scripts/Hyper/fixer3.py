#!/usr/bin/env python3
"""
Nested Except Block Fixer

Fixes the specific issue where except blocks are incorrectly nested inside try blocks,
causing SyntaxError: invalid syntax.

The problem:
    try:
        some_code()
        except Exception as e:  # ← This is INSIDE the try - INVALID!
            pass
    except Exception as e:      # ← This is correct position

Usage:
    python nested_except_fixer.py matcher50.py
"""

import sys
import os
import shutil
import ast
import re

def find_nested_except_blocks(lines):
    """Find except blocks that are incorrectly nested inside try blocks"""
    problematic_blocks = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Look for except blocks that contain the auto-generated warning
        if (stripped.startswith('except Exception as e:') and 
            i + 1 < len(lines) and
            'Unclosed try block exception' in lines[i + 1]):
            
            # Check if this except is inside a try block (incorrectly nested)
            line_indent = len(line) - len(line.lstrip())
            
            # Look backwards to find the try block
            for j in range(i - 1, max(0, i - 20), -1):
                prev_line = lines[j]
                prev_stripped = prev_line.strip()
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                
                if prev_stripped.startswith('try:'):
                    # Found a try block - check if our except is nested inside
                    expected_except_indent = prev_indent + 4
                    
                    if line_indent > expected_except_indent:
                        # This except is too indented - it's nested inside the try block
                        problematic_blocks.append({
                            'except_line': i,
                            'try_line': j,
                            'except_indent': line_indent,
                            'expected_indent': expected_except_indent,
                            'try_indent': prev_indent
                        })
                        break
                    elif line_indent == expected_except_indent:
                        # This except is at the right level - it's OK
                        break
    
    return problematic_blocks

def show_problem_detail(lines, problem_block):
    """Show the specific problem in detail"""
    try_line_num = problem_block['try_line'] + 1
    except_line_num = problem_block['except_line'] + 1
    
    print(f"\n🔍 NESTED EXCEPT PROBLEM DETAILS:")
    print(f"   Try block at line {try_line_num} (indent: {problem_block['try_indent']})")
    print(f"   Nested except at line {except_line_num} (indent: {problem_block['except_indent']})")
    print(f"   Expected except indent: {problem_block['expected_indent']}")
    
    print(f"\n📋 CODE CONTEXT:")
    start = max(0, problem_block['try_line'] - 3)
    end = min(len(lines), problem_block['except_line'] + 5)
    
    for i in range(start, end):
        line = lines[i]
        line_num = i + 1
        indent = len(line) - len(line.lstrip())
        
        if line_num == try_line_num:
            print(f"🔵 Line {line_num:3d}: ({indent:2d}) {line.rstrip()}")
        elif line_num == except_line_num:
            print(f"💥 Line {line_num:3d}: ({indent:2d}) {line.rstrip()}")
        else:
            print(f"   Line {line_num:3d}: ({indent:2d}) {line.rstrip()}")

def fix_nested_except_blocks(lines, problematic_blocks):
    """Remove incorrectly nested except blocks"""
    # Sort by line number in reverse order so we can delete without affecting indices
    problematic_blocks.sort(key=lambda x: x['except_line'], reverse=True)
    
    removed_lines = []
    
    for problem in problematic_blocks:
        except_line_idx = problem['except_line']
        
        # Find all lines that are part of this nested except block
        lines_to_remove = []
        current_idx = except_line_idx
        
        # Remove the except line itself
        if current_idx < len(lines) and lines[current_idx].strip().startswith('except Exception as e:'):
            lines_to_remove.append(current_idx)
            current_idx += 1
        
        # Remove indented lines that follow (part of the except block)
        base_indent = len(lines[except_line_idx]) - len(lines[except_line_idx].lstrip())
        
        while current_idx < len(lines):
            line = lines[current_idx]
            if line.strip() == "":  # Empty line
                current_idx += 1
                continue
            
            line_indent = len(line) - len(line.lstrip())
            
            # If this line is indented more than the except line, it's part of the except block
            if line_indent > base_indent:
                # Check if it's part of our problematic except block
                if ('logger.warning' in line and 'Unclosed try block' in line) or line.strip() == 'pass':
                    lines_to_remove.append(current_idx)
                    current_idx += 1
                else:
                    # This might be legitimate code, be careful
                    # Only remove if it's clearly auto-generated
                    if line.strip() == 'pass' and current_idx == except_line_idx + 2:
                        lines_to_remove.append(current_idx)
                    break
            else:
                break  # End of except block
        
        # Remove the lines (in reverse order within this block)
        for line_idx in reversed(lines_to_remove):
            removed_lines.append((line_idx + 1, lines[line_idx].rstrip()))
            del lines[line_idx]
    
    return lines, removed_lines

def validate_syntax(file_path):
    """Validate Python syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, "Syntax is valid"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python nested_except_fixer.py <python_file>")
        print("\nExample: python nested_except_fixer.py matcher50.py")
        print("\nThis fixes nested except blocks that cause SyntaxError.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        sys.exit(1)
    
    print("🔧 NESTED EXCEPT BLOCK FIXER")
    print("="*60)
    print(f"Target: {file_path}")
    print(f"Goal: Fix incorrectly nested except blocks")
    print("="*60)
    
    # Create backup
    backup_path = f"{file_path}.nested_backup"
    shutil.copy2(file_path, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    print(f"📄 File has {len(lines)} lines")
    
    # Check current syntax
    print(f"\n🔍 CHECKING CURRENT SYNTAX:")
    is_valid, message = validate_syntax(file_path)
    print(f"   Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"   Message: {message}")
    
    if is_valid:
        print("✅ File already has valid syntax!")
        return
    
    # Find nested except blocks
    problematic_blocks = find_nested_except_blocks(lines)
    
    if not problematic_blocks:
        print("❌ No nested except blocks found - may be a different syntax issue")
        return
    
    print(f"\n❌ Found {len(problematic_blocks)} nested except block(s):")
    
    for i, problem in enumerate(problematic_blocks, 1):
        print(f"\n🔍 PROBLEM {i}:")
        show_problem_detail(lines, problem)
    
    # Fix the problems
    print(f"\n🔧 APPLYING FIXES:")
    fixed_lines, removed_lines = fix_nested_except_blocks(lines, problematic_blocks)
    
    if removed_lines:
        print(f"🗑️ Removed {len(removed_lines)} problematic lines:")
        for line_num, content in removed_lines:
            print(f"   Line {line_num}: {content}")
    
    # Write fixed file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        print(f"✅ File updated with fixes")
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        shutil.copy2(backup_path, file_path)
        return
    
    # Validate after fix
    print(f"\n🔍 VALIDATING AFTER FIX:")
    is_valid, message = validate_syntax(file_path)
    print(f"   Status: {'✅ VALID' if is_valid else '❌ STILL INVALID'}")
    print(f"   Message: {message}")
    
    if is_valid:
        print(f"\n🚀 SUCCESS!")
        print(f"   ✅ Nested except blocks removed")
        print(f"   ✅ File should now compile")
        print(f"   💾 Backup: {backup_path}")
        
        print(f"\n💡 TEST THE FIX:")
        print(f"   python -m py_compile {file_path}")
        print(f"   ./runner.sh")
    else:
        print(f"\n⚠️ STILL HAS SYNTAX ERRORS")
        print(f"   There may be other issues in the file")
        
        # Try to show the remaining error location
        if "line" in message:
            try:
                error_line = int(message.split("line ")[1].split(":")[0])
                print(f"\n🔍 REMAINING ERROR AT LINE {error_line}:")
                
                start = max(0, error_line - 5)
                end = min(len(fixed_lines), error_line + 3)
                
                for i in range(start, end):
                    line_num = i + 1
                    marker = "💥" if line_num == error_line else "  "
                    print(f"{marker} Line {line_num:3d}: {fixed_lines[i].rstrip()}")
            except:
                pass

if __name__ == "__main__":
    main()