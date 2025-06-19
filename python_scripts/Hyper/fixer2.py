"""
Direct Line Remover - Simple Fix

Directly removes the problematic except block at line 104 and validates the fix.
No complex logic - just removes the bad lines and tests.

Usage:
    python direct_line_remover.py matcher50.py
"""

import sys
import os
import shutil
import ast

def show_problem_area(file_path: str, target_line: int = 104):
    """Show the problem area around the target line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    print(f"🔍 PROBLEM AREA AROUND LINE {target_line}:")
    print("="*60)
    
    start = max(0, target_line - 10)
    end = min(len(lines), target_line + 10)
    
    for i in range(start, end):
        line = lines[i]
        line_num = i + 1
        marker = "💥" if line_num == target_line else "  "
        print(f"{marker} Line {line_num:3d}: {line.rstrip()}")
    
    print("="*60)
    return lines

def remove_problematic_except_block(file_path: str, start_line: int = 104):
    """Remove the problematic except block starting at the specified line"""
    
    # Create backup first
    backup_path = f"{file_path}.direct_backup"
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
    
    # Check what's at the target line
    if start_line <= len(lines):
        target_line_content = lines[start_line - 1].strip()  # Convert to 0-based
        print(f"🎯 Line {start_line}: {target_line_content}")
        
        if target_line_content == "except Exception as e:":
            print(f"✅ Found problematic except block at line {start_line}")
            
            # Find how many lines to remove
            lines_to_remove = []
            current_line = start_line - 1  # Convert to 0-based index
            
            # Remove the except line
            if current_line < len(lines) and lines[current_line].strip() == "except Exception as e:":
                lines_to_remove.append(current_line)
                current_line += 1
            
            # Remove any indented lines that follow (part of the except block)
            while current_line < len(lines):
                line = lines[current_line]
                if line.strip() == "":  # Empty line
                    current_line += 1
                    continue
                elif line.startswith("    ") or line.startswith("\t"):  # Indented line
                    # Check if it's part of our problematic except block
                    if ("logger.warning" in line and "Unclosed try block" in line) or line.strip() == "pass":
                        lines_to_remove.append(current_line)
                        current_line += 1
                    else:
                        break
                else:
                    break  # Non-indented line, end of except block
            
            # Remove the lines (in reverse order to maintain indices)
            print(f"🗑️ Removing lines: {[i+1 for i in lines_to_remove]}")
            for line_index in reversed(lines_to_remove):
                print(f"   Removing line {line_index + 1}: {lines[line_index].rstrip()}")
                del lines[line_index]
            
            # Write the fixed file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"✅ Removed {len(lines_to_remove)} problematic lines")
                return True
            except Exception as e:
                print(f"❌ Error writing file: {e}")
                shutil.copy2(backup_path, file_path)  # Restore backup
                return False
        else:
            print(f"⚠️ Line {start_line} doesn't contain 'except Exception as e:'")
            print(f"   Content: {target_line_content}")
            return False
    else:
        print(f"❌ Line {start_line} is beyond file length ({len(lines)} lines)")
        return False

def validate_syntax(file_path: str):
    """Simple syntax validation"""
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
    """Main function - keep it simple"""
    if len(sys.argv) != 2:
        print("Usage: python direct_line_remover.py <python_file>")
        print("\nExample: python direct_line_remover.py matcher50.py")
        print("\nThis script directly removes the problematic except block at line 104.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        sys.exit(1)
    
    print("🔧 DIRECT LINE REMOVER - SIMPLE FIX")
    print("="*60)
    print(f"Target: {file_path}")
    print(f"Goal: Remove problematic except block at line 104")
    print("="*60)
    
    # Show the problem area first
    lines = show_problem_area(file_path, 104)
    if lines is None:
        sys.exit(1)
    
    # Check current syntax
    print(f"\n🔍 CHECKING CURRENT SYNTAX:")
    is_valid, message = validate_syntax(file_path)
    print(f"   Status: {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"   Message: {message}")
    
    if is_valid:
        print(f"✅ File already has valid syntax!")
        sys.exit(0)
    
    # Apply the fix
    print(f"\n🔧 APPLYING DIRECT FIX:")
    success = remove_problematic_except_block(file_path, 104)
    
    if success:
        # Validate after fix
        print(f"\n🔍 VALIDATING AFTER FIX:")
        is_valid, message = validate_syntax(file_path)
        print(f"   Status: {'✅ VALID' if is_valid else '❌ STILL INVALID'}")
        print(f"   Message: {message}")
        
        if is_valid:
            print(f"\n🚀 SUCCESS!")
            print(f"   ✅ Problematic except block removed")
            print(f"   ✅ File should now compile")
            print(f"   💾 Backup: {file_path}.direct_backup")
            
            print(f"\n💡 TEST THE FIX:")
            print(f"   python -m py_compile {file_path}")
            print(f"   ./runner.sh")
        else:
            print(f"\n⚠️ STILL HAS SYNTAX ERRORS")
            print(f"   There may be other syntax issues in the file")
            print(f"   Manual inspection may be needed")
            
            # Show the new problem area if there's still an error
            if "line" in message:
                try:
                    error_line = int(message.split("line ")[1].split(":")[0])
                    print(f"\n🔍 NEW ERROR LOCATION:")
                    show_problem_area(file_path, error_line)
                except:
                    pass
    else:
        print(f"\n❌ FAILED TO APPLY FIX")
        print(f"   Could not remove the problematic lines")
        print(f"   Original file restored from backup")

if __name__ == "__main__":
    main()