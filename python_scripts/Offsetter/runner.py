#!/usr/bin/env python3
"""
🚀 MULTI-GPU RUNNER SCRIPT 🚀
🔥 LAUNCHES SEPARATE GPU PROCESSES FOR OPTIMAL UTILIZATION 🔥
🌟 CLEAN SEPARATION OF WORK ACROSS MULTIPLE GPUS 🌟
"""

import argparse
import subprocess
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('multi_gpu_runner.log', mode='w')
        ]
    )
    return logging.getLogger('multi_gpu_runner')

def run_gpu_process(gpu_id: int, gpu_index: int, total_gpus: int, args: argparse.Namespace, script_path: str) -> Dict[str, Any]:
    """Run single GPU process and return results"""
    
    logger = logging.getLogger(f'GPU_{gpu_id}')
    
    # Build output filename
    input_path = Path(args.input_file)
    if args.output:
        output_file = f"{args.output}_gpu{gpu_id}.json"
    else:
        stem = input_path.stem
        output_file = input_path.parent / f"multi_gpu_{stem}_gpu{gpu_id}.json"
    
    # Build command
    cmd = [
        sys.executable, script_path,  # Use the found script path
        args.input_file,
        "--gpu-id", str(gpu_id),
        "--gpu-index", str(gpu_index),
        "--workers", str(args.workers),
        "--do-multiple-of", str(total_gpus),
        "--output", str(output_file),
        "--gpu-memory", str(args.gpu_memory),
        "--top-matches", str(args.top_matches),
        "--min-score", str(args.min_score),
        "--search-step", str(args.search_step),
        "--refinement-step", str(args.refinement_step),
        "--search-range", str(args.search_range)
    ]
    
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    if args.debug:
        cmd.append("--debug")
    
    if args.gpu_debug:
        cmd.append("--gpu-debug")
    
    logger.info(f"🚀 Starting GPU {gpu_id} process with {args.workers} workers...")
    logger.info(f"📜 Script: {script_path}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    # Run process with real-time output capture
    start_time = time.time()
    try:
        # Use Popen for better control and real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture output in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Log important lines
                if any(keyword in output for keyword in ['ERROR', 'WARNING', '💀', '❌', '⚠️', '✅']):
                    logger.info(f"GPU {gpu_id}: {output.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        processing_time = time.time() - start_time
        
        # Capture any remaining output
        try:
            remaining_output, _ = process.communicate(timeout=5)
            if remaining_output:
                output_lines.extend(remaining_output.strip().split('\n'))
        except:
            pass
        
        if return_code == 0:
            logger.info(f"✅ GPU {gpu_id} completed successfully in {processing_time:.1f}s")
            
            # Check if output file was actually created
            if not Path(output_file).exists():
                logger.error(f"❌ GPU {gpu_id}: Process succeeded but no output file created!")
                logger.error(f"Expected file: {output_file}")
                logger.error("Last 10 lines of output:")
                for line in output_lines[-10:]:
                    logger.error(f"  {line}")
                
                return {
                    'gpu_id': gpu_id,
                    'success': False,
                    'error': f"No output file created: {output_file}",
                    'processing_time': processing_time,
                    'workers_used': args.workers,
                    'output_lines': output_lines[-20:]  # Last 20 lines for debugging
                }
            
            # Load and return stats
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                stats = data.get('single_gpu_processing_info', {})
                return {
                    'gpu_id': gpu_id,
                    'success': True,
                    'output_file': str(output_file),
                    'processing_time': processing_time,
                    'workers_used': args.workers,
                    'matches_processed': stats.get('matches_processed', 0),
                    'matches_successful': stats.get('matches_successful', 0),
                    'success_rate': stats.get('success_rate', 0),
                    'processing_rate': stats.get('processing_rate', 0)
                }
            except Exception as e:
                logger.warning(f"⚠️  GPU {gpu_id}: Could not load result stats: {e}")
                return {
                    'gpu_id': gpu_id,
                    'success': True,
                    'output_file': str(output_file),
                    'processing_time': processing_time,
                    'workers_used': args.workers,
                    'matches_processed': 0,
                    'matches_successful': 0,
                    'error': f"Could not load stats: {e}"
                }
        else:
            logger.error(f"❌ GPU {gpu_id} failed with exit code {return_code}")
            logger.error("Process output:")
            for line in output_lines[-20:]:  # Last 20 lines
                logger.error(f"  {line}")
            
            return {
                'gpu_id': gpu_id,
                'success': False,
                'error': f"Exit code {return_code}",
                'processing_time': processing_time,
                'workers_used': args.workers,
                'output_lines': output_lines[-20:]
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ GPU {gpu_id} timed out after 2 hours")
        try:
            process.kill()
        except:
            pass
        return {
            'gpu_id': gpu_id,
            'success': False,
            'error': "Process timed out after 2 hours",
            'processing_time': processing_time,
            'workers_used': args.workers
        }
    except Exception as e:
        logger.error(f"❌ GPU {gpu_id} unexpected error: {e}")
        return {
            'gpu_id': gpu_id,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'workers_used': args.workers
        }

def merge_results(results: List[Dict], input_file: str, output_prefix: str) -> str:
    """Merge results from multiple GPU outputs"""
    
    logger = logging.getLogger('merger')
    
    # Find successful results
    successful_results = [r for r in results if r['success'] and 'output_file' in r]
    
    if len(successful_results) == 0:
        logger.error("❌ No successful results to merge")
        return None
    
    logger.info(f"🔄 Merging {len(successful_results)} GPU results...")
    
    # Load base data from original input
    with open(input_file, 'r') as f:
        merged_data = json.load(f)
    
    # Collect all results by video-gpx key
    all_processed_results = {}
    total_stats = {
        'matches_processed': 0,
        'matches_successful': 0,
        'total_processing_time': 0,
        'gpu_stats': {}
    }
    
    # Process each GPU's results
    for result in successful_results:
        gpu_id = result['gpu_id']
        output_file = result['output_file']
        
        try:
            with open(output_file, 'r') as f:
                gpu_data = json.load(f)
            
            # Extract processed matches
            for video_path, video_data in gpu_data.get('results', {}).items():
                for match in video_data.get('matches', []):
                    # Check if this match was actually processed by this GPU
                    if match.get('actual_processing_gpu') == gpu_id:
                        gpx_path = match.get('path', '')
                        key = (video_path, gpx_path)
                        all_processed_results[key] = match
            
            # Collect stats
            gpu_stats = gpu_data.get('single_gpu_processing_info', {})
            total_stats['matches_processed'] += gpu_stats.get('matches_processed', 0)
            total_stats['matches_successful'] += gpu_stats.get('matches_successful', 0)
            total_stats['total_processing_time'] += gpu_stats.get('processing_time_seconds', 0)
            total_stats['gpu_stats'][gpu_id] = gpu_stats
            
        except Exception as e:
            logger.warning(f"⚠️  Could not process GPU {gpu_id} results: {e}")
    
    # Merge results back into original structure
    merged_results = {}
    for video_path, video_data in merged_data.get('results', {}).items():
        merged_video_data = video_data.copy()
        merged_matches = []
        
        for match in video_data.get('matches', []):
            gpx_path = match.get('path', '')
            key = (video_path, gpx_path)
            
            if key in all_processed_results:
                # Use processed result
                merged_matches.append(all_processed_results[key])
            else:
                # Use original match
                merged_matches.append(match)
        
        merged_video_data['matches'] = merged_matches
        merged_results[video_path] = merged_video_data
    
    merged_data['results'] = merged_results
    
    # Add combined processing info
    merged_data['multi_gpu_processing_info'] = {
        'multi_gpu_mode': True,
        'total_gpus_used': len(successful_results),
        'gpu_ids': [r['gpu_id'] for r in successful_results],
        'total_matches_processed': total_stats['matches_processed'],
        'total_matches_successful': total_stats['matches_successful'],
        'overall_success_rate': total_stats['matches_successful'] / total_stats['matches_processed'] if total_stats['matches_processed'] > 0 else 0,
        'total_processing_time_seconds': total_stats['total_processing_time'],
        'parallel_processing_time_seconds': max(r['processing_time'] for r in successful_results),
        'speedup_factor': total_stats['total_processing_time'] / max(r['processing_time'] for r in successful_results),
        'gpu_individual_stats': total_stats['gpu_stats'],
        'processed_at': time.strftime('%Y-%m-%dT%H:%M:%S')
    }
    
    # Save merged results
    merged_file = f"{output_prefix}_merged.json"
    with open(merged_file, 'w') as f:
        json.dump(merged_data, f, indent=2, default=str)
    
    logger.info(f"✅ Merged results saved to {merged_file}")
    return merged_file

def main():
    parser = argparse.ArgumentParser(
        description='🚀 Multi-GPU Video Synchronization Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Basic dual GPU run
    python multi_gpu_runner.py complete_turbo_360_report_ramcache.json

    # Custom GPUs and settings  
    python multi_gpu_runner.py input.json --gpu-ids 0,1,2 --gpu-memory 8.0

    # Testing with limits
    python multi_gpu_runner.py input.json --limit 20 --debug --no-merge

MULTI-GPU WORK DISTRIBUTION:
    GPU 0 processes: files 0, 2, 4, 6, 8, ...  (even indices)
    GPU 1 processes: files 1, 3, 5, 7, 9, ...  (odd indices)  
    GPU 2 processes: files 2, 5, 8, 11, ...    (every 3rd starting at 2)
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file with matches')
    parser.add_argument('-o', '--output', help='Output file prefix')
    parser.add_argument('--gpu-ids', default='0,1', help='Comma-separated GPU IDs (default: "0,1")')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers per GPU (default: 1)')
    parser.add_argument('--gpu-memory', type=float, default=14.0, help='GPU memory limit per GPU in GB (default: 14.0)')
    parser.add_argument('--top-matches', type=int, default=4, help='Maximum matches per video (default: 4)')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score threshold (default: 0.3)')
    parser.add_argument('--limit', type=int, help='Limit total matches for testing')
    parser.add_argument('--search-step', type=float, default=0.05, help='Search step size in seconds (default: 0.05)')
    parser.add_argument('--refinement-step', type=float, default=0.01, help='Refinement step size (default: 0.01)')
    parser.add_argument('--search-range', type=float, default=90.0, help='Max search range in seconds (default: 90.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--gpu-debug', action='store_true', help='Enable detailed GPU context debugging')
    parser.add_argument('--no-merge', action='store_true', help="Don't merge results (keep separate files)")
    parser.add_argument('--sequential', action='store_true', help='Run GPUs sequentially instead of parallel (for debugging)')
    parser.add_argument('--test-script', action='store_true', help='Test script execution and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 MULTI-GPU VIDEO SYNCHRONIZATION RUNNER")
    logger.info("=" * 60)
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"💀 Input file not found: {input_file}")
        sys.exit(1)
    
    # Check for required script
    script_path = Path("single_gpu_offsetter.py")
    if not script_path.exists():
        # Try alternative names
        alternative_names = ["offsetter27.py", "offsetter26.py", "offsetter25.py", "offsetter24.py", "offsetter23.py"]
        script_found = False
        for alt_name in alternative_names:
            if Path(alt_name).exists():
                script_path = Path(alt_name)
                script_found = True
                logger.info(f"🔧 Using {alt_name} instead of single_gpu_offsetter.py")
                break
        
        if not script_found:
            logger.error(f"💀 Required script not found. Tried:")
            logger.error(f"   - single_gpu_offsetter.py")
            for alt_name in alternative_names:
                logger.error(f"   - {alt_name}")
            logger.error("Please ensure the single GPU processor script is in the current directory")
            sys.exit(1)
    
    # Test script execution if requested
    if args.test_script:
        logger.info(f"🧪 Testing script: {script_path}")
        
        # Test 1: Help command
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info("✅ Script help command works")
            else:
                logger.error(f"❌ Script help failed: {result.stderr}")
        except Exception as e:
            logger.error(f"❌ Script help test error: {e}")
        
        # Test 2: Quick data test
        logger.info("🧪 Testing with minimal parameters...")
        test_cmd = [
            sys.executable, str(script_path),
            str(input_file),
            "--gpu-id", "0",
            "--workers", "1",
            "--limit", "1",
            "--min-score", "0.0",
            "--output", "test_output.json",
            "--debug"
        ]
        
        try:
            logger.info(f"Test command: {' '.join(test_cmd)}")
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info("✅ Basic script execution works")
                if Path("test_output.json").exists():
                    logger.info("✅ Output file created successfully")
                    # Clean up
                    Path("test_output.json").unlink()
                else:
                    logger.warning("⚠️  Script ran but no output file created")
            else:
                logger.error(f"❌ Script execution failed:")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        except Exception as e:
            logger.error(f"❌ Script execution test error: {e}")
        
        logger.info("🧪 Script testing complete")
        sys.exit(0)
    
    # Parse GPU IDs
    try:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    except ValueError:
        logger.error(f"💀 Invalid GPU IDs: {args.gpu_ids}")
        sys.exit(1)
    
    num_gpus = len(gpu_ids)
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = f"multi_gpu_{input_file.stem}"
    
    logger.info(f"📁 Input file: {input_file}")
    logger.info(f"🎮 GPUs: {gpu_ids} ({num_gpus} total)")
    logger.info(f"👥 Workers: {args.workers} per GPU")
    logger.info(f"💾 GPU memory: {args.gpu_memory}GB per GPU")
    logger.info(f"🎯 Top matches: {args.top_matches} per video")
    logger.info(f"📊 Min score: {args.min_score}")
    if args.limit:
        logger.info(f"🔢 Limit: {args.limit} matches")
    logger.info(f"📤 Output prefix: {output_prefix}")
    logger.info("=" * 60)
    
    # Run GPU processes
    start_time = time.time()
    
    if args.sequential:
        logger.info("🔄 Running GPUs sequentially...")
        results = []
        for i, gpu_id in enumerate(gpu_ids):
            result = run_gpu_process(gpu_id, i, num_gpus, args, str(script_path))
            results.append(result)
    else:
        logger.info(f"🚀 Launching {num_gpus} GPU processes in parallel...")
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            # Submit all jobs
            futures = []
            for i, gpu_id in enumerate(gpu_ids):
                future = executor.submit(run_gpu_process, gpu_id, i, num_gpus, args, str(script_path))
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"💀 Process execution failed: {e}")
                    results.append({
                        'success': False,
                        'error': f"Process execution failed: {e}"
                    })
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    logger.info("")
    logger.info("🎉 All GPU processes completed!")
    logger.info("=" * 60)
    logger.info(f"✅ Successful GPUs: {len(successful_results)}/{num_gpus}")
    logger.info(f"⏱️  Total wall time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    if successful_results:
        total_processed = sum(r.get('matches_processed', 0) for r in successful_results)
        total_successful = sum(r.get('matches_successful', 0) for r in successful_results)
        total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
        
        logger.info(f"📊 Total matches processed: {total_processed}")
        logger.info(f"📈 Total successful: {total_successful}")
        logger.info(f"📊 Overall success rate: {total_successful/total_processed*100:.1f}%" if total_processed > 0 else "0%")
        logger.info(f"⚡ Cumulative processing time: {total_processing_time:.1f}s")
        logger.info(f"🚀 Speedup factor: {total_processing_time/total_time:.1f}x")
        
        # Individual GPU stats
        logger.info("")
        logger.info("📊 Individual GPU performance:")
        for result in successful_results:
            gpu_id = result['gpu_id']
            workers = result.get('workers_used', 1)
            processed = result.get('matches_processed', 0)
            successful = result.get('matches_successful', 0)
            time_taken = result.get('processing_time', 0)
            rate = result.get('processing_rate', 0)
            
            logger.info(f"   GPU {gpu_id} ({workers}w): {processed} processed, {successful} successful, "
                       f"{time_taken:.1f}s, {rate:.2f}/s")
    
    if failed_results:
        logger.error("")
        logger.error("❌ Failed GPUs:")
        for result in failed_results:
            gpu_id = result.get('gpu_id', 'unknown')
            error = result.get('error', 'Unknown error')
            logger.error(f"   GPU {gpu_id}: {error}")
            
            # Show debug output if available
            if 'output_lines' in result:
                logger.error(f"   Last output lines:")
                for line in result['output_lines'][-5:]:  # Last 5 lines
                    logger.error(f"     {line}")
        
        # Additional debugging
        logger.error("")
        logger.error("🔍 DEBUGGING INFORMATION:")
        logger.error(f"   Script used: {script_path}")
        logger.error(f"   Script exists: {script_path.exists()}")
        logger.error(f"   Input file: {input_file}")
        logger.error(f"   Input file exists: {input_file.exists()}")
        logger.error(f"   Current directory: {Path.cwd()}")
        logger.error(f"   Available Python files: {[f.name for f in Path('.').glob('*.py')]}")
        
        # Test basic command
        logger.error("   Testing basic script execution:")
        try:
            test_result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if test_result.returncode == 0:
                logger.error("     ✅ Script help command works")
            else:
                logger.error(f"     ❌ Script help failed: {test_result.stderr}")
        except Exception as e:
            logger.error(f"     ❌ Script test error: {e}")
    
    # Merge results if requested and successful
    if not args.no_merge and len(successful_results) > 0:
        logger.info("")
        try:
            merged_file = merge_results(successful_results, str(input_file), output_prefix)
            if merged_file:
                logger.info(f"🔗 Merged results: {merged_file}")
        except Exception as e:
            logger.error(f"❌ Failed to merge results: {e}")
    
    # List output files
    logger.info("")
    logger.info("📁 Output files:")
    for result in successful_results:
        if 'output_file' in result:
            logger.info(f"   {Path(result['output_file']).name}")
    
    logger.info("")
    logger.info("🎉 Multi-GPU processing complete!")
    logger.info("=" * 60)
    
    # Exit with appropriate code
    if len(successful_results) == 0:
        sys.exit(1)
    elif len(failed_results) > 0:
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()