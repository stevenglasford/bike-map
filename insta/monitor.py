#!/usr/bin/env python3
"""
Insta360 Processing Monitor
Real-time monitoring of video processing progress
"""

import json
import time
import os
import sys

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

def load_progress():
    """Load progress data from report file"""
    try:
        with open('/logs/progress_report.json', 'r') as f:
            return json.load(f)
    except:
        return None

def create_status_display():
    """Create status display table"""
    progress = load_progress()
    if not progress:
        if RICH_AVAILABLE:
            return Table(title="No progress data available")
        else:
            return "No progress data available"
    
    stats = progress['stats']
    
    if RICH_AVAILABLE:
        table = Table(title="🎥 Insta360 Processing Status")
        table.add_column("Camera", style="cyan")
        table.add_column("Total", style="white")
        table.add_column("Processed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Remaining", style="yellow")
        table.add_column("Success %", style="magenta")
        
        for camera, data in progress.get('cameras', {}).items():
            table.add_row(
                camera,
                str(data['total']),
                str(data['processed']),
                str(data['failed']),
                str(data['remaining']),
                f"{data['success_rate']:.1f}%"
            )
        
        return table
    else:
        # Simple text version
        output = "\n🎥 Insta360 Processing Status\n"
        output += "=" * 40 + "\n"
        
        for camera, data in progress.get('cameras', {}).items():
            output += f"{camera:10} | "
            output += f"Total: {data['total']:3} | "
            output += f"Processed: {data['processed']:3} | "
            output += f"Failed: {data['failed']:2} | "
            output += f"Remaining: {data['remaining']:3} | "
            output += f"Success: {data['success_rate']:5.1f}%\n"
        
        return output

def monitor():
    """Main monitoring loop"""
    if RICH_AVAILABLE:
        with Live(create_status_display(), refresh_per_second=1) as live:
            while True:
                live.update(create_status_display())
                time.sleep(5)
    else:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(create_status_display())
            time.sleep(5)

def show_summary():
    """Show final summary"""
    progress = load_progress()
    if not progress:
        print("No progress data found")
        return
    
    stats = progress['stats']
    print(f"\nFinal Summary:")
    print(f"Total videos: {stats.get('total_found', 0)}")
    print(f"Processed: {stats.get('processed', 0)}")
    print(f"Failed: {stats.get('failed', 0)}")
    print(f"Success rate: {(stats.get('processed', 0) / max(stats.get('total_found', 1), 1)) * 100:.1f}%")
    
    print("\nBy camera:")
    for camera, data in progress.get('cameras', {}).items():
        print(f"  {camera}: {data['processed']}/{data['total']} ({data['success_rate']:.1f}%)")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "summary":
            show_summary()
            return
        elif sys.argv[1] == "once":
            print(create_status_display())
            return
    
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    main()