#!/usr/bin/env python3
"""
Quick Test Script for GPX Validation

Run this BEFORE modifying your main script to see the impact:
    python test_gpx_validation.py ~/penis/panoramics/playground/

This will show you exactly what to expect from the integration.
"""

import os
import sys
import time
from pathlib import Path
import json

# Import the production validator
from production_gpx_validator import (
    ProductionGPXValidator, 
    GPXValidationConfig, 
    ValidationLevel
)

def test_validation_levels(directory: str):
    """Test all validation levels and show expected results"""
    
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    gpx_files = list(directory.glob("*.gpx"))
    
    if not gpx_files:
        print(f"❌ No GPX files found in {directory}")
        return
    
    print(f"🔍 Testing GPX validation on {len(gpx_files)} files from {directory}")
    print(f"📁 Total GPX files found: {len(gpx_files)}")
    print("=" * 80)
    
    # Test each validation level
    levels = [
        ValidationLevel.STRICT,
        ValidationLevel.MODERATE, 
        ValidationLevel.LENIENT
    ]
    
    results_summary = {}
    
    for level in levels:
        print(f"\n🔍 Testing {level.value.upper()} validation...")
        
        # Configure validation
        config = GPXValidationConfig.get_preset(level)
        config.enable_parallel_processing = True
        config.max_workers = 4
        config.save_diagnostics = False  # Don't save for testing
        
        # Run validation
        validator = ProductionGPXValidator(config)
        start_time = time.time()
        
        results, summary = validator.validate_directory(directory)
        
        elapsed = time.time() - start_time
        
        # Store results
        results_summary[level.value] = {
            'total_files': summary['total_files'],
            'valid_files': summary['valid_files'],
            'validation_rate': summary['validation_rate'],
            'avg_quality_score': summary['avg_quality_score'],
            'processing_time': elapsed,
            'rejection_reasons': dict(summary['rejection_reasons'])
        }
        
        # Print quick summary
        print(f"   ✅ Valid files: {summary['valid_files']:,} ({summary['validation_rate']:.1%})")
        print(f"   ❌ Invalid files: {summary['invalid_files']:,}")
        print(f"   📊 Avg quality: {summary['avg_quality_score']:.3f}")
        print(f"   ⏱️  Processing time: {elapsed:.1f}s")
        
        # Show top rejection reasons
        if summary['rejection_reasons']:
            top_reasons = list(summary['rejection_reasons'].most_common(3))
            print(f"   🚫 Top rejections: {', '.join([f'{r}({c})' for r, c in top_reasons])}")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("📊 VALIDATION LEVEL COMPARISON")
    print("=" * 80)
    
    current_valid = 10  # From their production report
    
    for level, data in results_summary.items():
        improvement = data['valid_files'] / current_valid if current_valid > 0 else 0
        correlations = 55 * data['valid_files']  # 55 videos
        
        print(f"{level.upper():>10}: {data['valid_files']:>6,} files ({data['validation_rate']:>6.1%}) "
              f"→ {correlations:>8,} correlations (📈 {improvement:.0f}x improvement)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    moderate_rate = results_summary['moderate']['validation_rate']
    lenient_rate = results_summary['lenient']['validation_rate']
    
    if moderate_rate >= 0.4:
        print(f"   🎯 RECOMMENDED: Use MODERATE validation")
        print(f"      • Good balance: {results_summary['moderate']['valid_files']:,} files ({moderate_rate:.1%})")
        print(f"      • Quality score: {results_summary['moderate']['avg_quality_score']:.3f}")
        print(f"      • Total correlations: {55 * results_summary['moderate']['valid_files']:,}")
    elif lenient_rate >= 0.6:
        print(f"   ⚠️  RECOMMENDED: Use LENIENT validation")
        print(f"      • Higher inclusion: {results_summary['lenient']['valid_files']:,} files ({lenient_rate:.1%})")
        print(f"      • Quality score: {results_summary['lenient']['avg_quality_score']:.3f}")
        print(f"      • Total correlations: {55 * results_summary['lenient']['valid_files']:,}")
    else:
        print(f"   🔧 RECOMMENDED: Use CUSTOM validation")
        print(f"      • Run: python gpx_diagnostic.py {directory}")
        print(f"      • Use recommended thresholds for higher pass rate")
    
    # Integration preview
    print(f"\n🚀 INTEGRATION PREVIEW:")
    print(f"   Current system: 10 valid GPX files → 550 correlations")
    best_level = max(results_summary.keys(), 
                    key=lambda x: results_summary[x]['validation_rate'])
    best_data = results_summary[best_level]
    print(f"   With {best_level}: {best_data['valid_files']:,} valid GPX files → {55 * best_data['valid_files']:,} correlations")
    improvement = (55 * best_data['valid_files']) / 550 if 550 > 0 else 0
    print(f"   📈 Expected improvement: {improvement:.0f}x more correlations!")
    
    # Save results for reference
    output_file = "validation_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results_summary

def show_integration_commands(directory: str, results: dict):
    """Show the exact commands to use for integration"""
    
    print(f"\n🔧 INTEGRATION COMMANDS:")
    print("=" * 50)
    
    # Find best validation level
    best_level = max(results.keys(), key=lambda x: results[x]['validation_rate'])
    
    print(f"1. First, run diagnostic:")
    print(f"   python gpx_diagnostic.py {directory}")
    
    print(f"\n2. Test with recommended level:")
    print(f"   ./runner.sh --gpx-validation {best_level}")
    
    print(f"\n3. If that works well, integrate into matcher43.py:")
    print(f"   # Replace GPS processor initialization with:")
    print(f"   gpx_config = GPXValidationConfig.get_preset(ValidationLevel.{best_level.upper()})")
    print(f"   gpx_validator = ProductionGPXValidator(gpx_config)")
    
    print(f"\n4. Expected results:")
    best_data = results[best_level]
    print(f"   • {best_data['valid_files']:,} valid GPX files ({best_data['validation_rate']:.1%})")
    print(f"   • {55 * best_data['valid_files']:,} total correlations")
    print(f"   • {best_data['avg_quality_score']:.3f} average quality score")

def main():
    """Main test function"""
    
    if len(sys.argv) != 2:
        print("Usage: python test_gpx_validation.py <gpx_directory>")
        print("Example: python test_gpx_validation.py ~/penis/panoramics/playground/")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    print("🧪 GPX Validation Test Suite")
    print("=" * 50)
    print("This will test different validation levels on your data")
    print("WITHOUT modifying your main script.\n")
    
    try:
        results = test_validation_levels(directory)
        
        if results:
            show_integration_commands(directory, results)
            
        print(f"\n✅ Test complete! Ready for integration.")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"💡 Make sure production_gpx_validator.py is in the same directory")

if __name__ == "__main__":
    main()