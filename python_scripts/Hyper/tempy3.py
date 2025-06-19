# =====================================================================

# STEP 1: Add imports at the top of matcher43.py (after existing imports)

# =====================================================================

# Add these imports after your existing imports

from production_gpx_validator import (
ProductionGPXValidator,
GPXValidationConfig,
ValidationLevel
)

# =====================================================================

# STEP 2: Modify the Enhanced360ProcessingConfig class

# =====================================================================

@dataclass
class Enhanced360ProcessingConfig:
“”“Enhanced configuration optimized for 360° and panoramic videos”””
# … keep all your existing parameters …

```
# Add GPX validation configuration
gpx_validation_level: ValidationLevel = ValidationLevel.MODERATE
gpx_custom_config: Optional[GPXValidationConfig] = None
enable_gpx_diagnostics: bool = True
gpx_diagnostics_file: str = "gpx_validation.db"
```

# =====================================================================

# STEP 3: Create Enhanced GPS Processor with Validation

# =====================================================================

class EnhancedGPSProcessor(AdvancedGPSProcessor):
“”“Enhanced GPS processor with production validation”””

```
def __init__(self, config: Enhanced360ProcessingConfig):
    super().__init__(config)
    
    # Initialize production validator
    if config.gpx_custom_config:
        gpx_config = config.gpx_custom_config
    else:
        gpx_config = GPXValidationConfig.get_preset(config.gpx_validation_level)
    
    # Configure validation settings
    gpx_config.save_diagnostics = config.enable_gpx_diagnostics
    gpx_config.diagnostics_file = config.gpx_diagnostics_file
    gpx_config.enable_parallel_processing = True
    gpx_config.max_workers = min(4, mp.cpu_count())
    
    self.validator = ProductionGPXValidator(gpx_config)
    self.validation_stats = {"total": 0, "valid": 0, "invalid": 0}
    
    logger.info(f"🔍 GPX Validation Level: {gpx_config.level.value.upper()}")

def process_gpx_batch_with_validation(self, gpx_files: List[str]) -> Dict[str, Dict]:
    """Process GPX files with production validation"""
    logger.info(f"🔍 Validating {len(gpx_files)} GPX files...")
    
    valid_gpx_data = {}
    
    # Get directory and validate all files
    if gpx_files:
        directory = os.path.dirname(gpx_files[0])
        validation_results, summary = self.validator.validate_directory(
            directory, pattern="*.gpx"
        )
        
        # Print validation summary
        self.validator.print_validation_report(validation_results, summary)
        
        # Create lookup for validation results
        result_lookup = {}
        for result in validation_results:
            # Extract filename from full path in validation results
            for gpx_file in gpx_files:
                if os.path.basename(gpx_file) in str(result.metadata.get('filename', '')):
                    result_lookup[gpx_file] = result
                    break
        
        # Process only valid files
        for gpx_file in gpx_files:
            self.validation_stats["total"] += 1
            
            # Check if file passed validation
            validation_result = None
            for result in validation_results:
                if os.path.basename(gpx_file) in str(result):
                    validation_result = result
                    break
            
            if validation_result and validation_result.is_valid:
                self.validation_stats["valid"] += 1
                
                # Process the validated file
                gpx_data = self.process_gpx_enhanced(gpx_file)
                if gpx_data:
                    # Add validation metadata
                    gpx_data['validation_quality'] = validation_result.quality_score
                    gpx_data['validation_warnings'] = validation_result.warnings
                    gpx_data['validation_metadata'] = validation_result.metadata
                    valid_gpx_data[gpx_file] = gpx_data
                    
                    logger.debug(f"✅ Processed {os.path.basename(gpx_file)} "
                               f"(quality: {validation_result.quality_score:.3f})")
            else:
                self.validation_stats["invalid"] += 1
                if validation_result:
                    logger.debug(f"❌ Rejected {os.path.basename(gpx_file)}: "
                               f"{validation_result.rejection_reason}")
    
    validation_rate = (self.validation_stats["valid"] / 
                      self.validation_stats["total"] if self.validation_stats["total"] > 0 else 0)
    
    logger.info(f"✅ GPX Validation Complete: {self.validation_stats['valid']} valid, "
               f"{self.validation_stats['invalid']} invalid "
               f"({validation_rate:.1%} success rate)")
    
    return valid_gpx_data
```

# =====================================================================

# STEP 4: Modify the main() function

# =====================================================================

def main():
“”“Enhanced main function with production GPX validation”””
parser = argparse.ArgumentParser(
description=“Enhanced High-Accuracy Video-GPX Correlation System (360° Optimized)”,
formatter_class=argparse.RawDescriptionHelpFormatter
)

```
# ... keep all your existing arguments ...

# Add GPX validation arguments
parser.add_argument("--gpx-validation", 
                   choices=['strict', 'moderate', 'lenient', 'custom'],
                   default='moderate',
                   help="GPX validation level")
parser.add_argument("--gpx-diagnostics", action='store_true', 
                   help="Enable detailed GPX validation diagnostics")
parser.add_argument("--gpx-diagnose-only", action='store_true',
                   help="Only run GPX diagnostics, don't process videos")

args = parser.parse_args()

# Run GPX diagnostics if requested
if args.gpx_diagnose_only:
    logger.info("🔍 Running GPX diagnostics only...")
    os.system(f"python gpx_diagnostic.py {args.directory}")
    return

# Configure GPX validation level
if args.gpx_validation == 'strict':
    gpx_validation_level = ValidationLevel.STRICT
elif args.gpx_validation == 'moderate':
    gpx_validation_level = ValidationLevel.MODERATE
elif args.gpx_validation == 'lenient':
    gpx_validation_level = ValidationLevel.LENIENT
else:  # custom
    gpx_validation_level = ValidationLevel.CUSTOM

# Create enhanced configuration
config = Enhanced360ProcessingConfig(
    max_frames=args.max_frames,
    target_size=tuple(args.video_size),
    sample_rate=args.sample_rate,
    powersafe=args.powersafe,
    strict=args.strict,
    enable_preprocessing=args.enable_preprocessing,
    ram_cache_gb=args.ram_cache,
    use_optical_flow=not args.disable_optical_flow,
    use_pretrained_features=not args.disable_pretrained_cnn,
    use_attention_mechanism=not args.disable_attention,
    use_ensemble_matching=not args.disable_ensemble,
    detect_360_video=not args.disable_360_detection,
    enable_spherical_processing=not args.disable_spherical_processing,
    enable_tangent_plane_processing=not args.disable_tangent_planes,
    
    # GPX validation configuration
    gpx_validation_level=gpx_validation_level,
    enable_gpx_diagnostics=args.gpx_diagnostics,
    gpx_diagnostics_file=os.path.join(args.output, "gpx_validation.db")
)

# ... keep your existing initialization code ...

logger.info("🚀 Starting Enhanced High-Accuracy Video-GPX Correlation System")
logger.info("🌐 360° & Panoramic Video Optimized")
logger.info(f"🔍 GPX Validation: {gpx_validation_level.value.upper()}")

# Initialize enhanced GPS processor
gps_processor = EnhancedGPSProcessor(config)

# ... rest of your existing main() function ...
```

# =====================================================================

# STEP 5: Modify your video processing loop

# =====================================================================

def process_videos_and_gpx(video_files, gpx_files, config, args):
“”“Modified processing function with enhanced GPX validation”””

```
# Initialize processors
gps_processor = EnhancedGPSProcessor(config)

# Process GPX files with validation
logger.info(f"🔍 Processing {len(gpx_files)} GPX files with validation...")
valid_gpx_data = gps_processor.process_gpx_batch_with_validation(gpx_files)

logger.info(f"✅ {len(valid_gpx_data)} valid GPX files ready for correlation")

if len(valid_gpx_data) == 0:
    logger.error("❌ No valid GPX files found! Check validation settings.")
    logger.info("💡 Try running with --gpx-diagnose-only to analyze your data")
    logger.info("💡 Or use --gpx-validation lenient for more permissive validation")
    return {}

# Continue with your existing video processing...
results = {}

for video_file in video_files:
    logger.info(f"🎥 Processing video: {os.path.basename(video_file)}")
    
    # Extract video features (your existing code)
    video_features = extract_video_features(video_file, config)
    
    if not video_features:
        continue
    
    # Correlate with valid GPX files
    video_matches = []
    
    for gpx_file, gpx_data in valid_gpx_data.items():
        # Your existing correlation code
        similarity = compute_similarity(video_features, gpx_data)
        
        if similarity['combined'] > 0.3:  # Your threshold
            match = {
                'path': gpx_file,
                'combined_score': similarity['combined'],
                'quality': similarity['quality'],
                'validation_quality': gpx_data['validation_quality'],
                'validation_warnings': len(gpx_data['validation_warnings']),
                # ... your existing match data ...
            }
            video_matches.append(match)
    
    # Sort by combined score including validation quality
    video_matches.sort(
        key=lambda x: (x['combined_score'] * x['validation_quality']), 
        reverse=True
    )
    
    if video_matches:
        results[video_file] = {'matches': video_matches[:5]}  # Top 5

return results
```

# =====================================================================

# STEP 6: Add custom validation configuration (if needed)

# =====================================================================

def create_custom_gpx_config() -> GPXValidationConfig:
“”“Create custom GPX validation config based on your data analysis”””

```
# Run this after analyzing your data with gpx_diagnostic.py
# Replace these values with recommendations from the diagnostic

return GPXValidationConfig(
    level=ValidationLevel.CUSTOM,
    min_points=10,                    # Adjust based on diagnostic
    min_duration_seconds=30,          # Adjust based on diagnostic  
    min_distance_meters=50,           # Adjust based on diagnostic
    max_speed_kmh=300,               # Adjust based on diagnostic
    require_timestamps=True,
    allow_synthetic_timestamps=True,  # Helps with missing timestamps
    min_quality_score=0.2,           # Lower for more permissive
    max_missing_timestamps_ratio=0.3, # Allow more missing timestamps
    max_duplicate_points_ratio=0.5,   # Allow more duplicates
    max_static_points_ratio=0.9,      # Allow mostly static points
    enable_statistical_validation=False,  # Disable for more permissive
    enable_parallel_processing=True,
    max_workers=4,
    save_diagnostics=True
)
```

# =====================================================================

# STEP 7: Add validation reporting to your results

# =====================================================================

def save_enhanced_results(results, gps_processor, output_file):
“”“Save results with validation statistics”””

```
# Your existing results saving code
enhanced_results = {
    "processing_info": {
        # ... your existing processing info ...
        "gpx_validation": {
            "total_gpx_files": gps_processor.validation_stats["total"],
            "valid_gpx_files": gps_processor.validation_stats["valid"], 
            "invalid_gpx_files": gps_processor.validation_stats["invalid"],
            "validation_rate": (gps_processor.validation_stats["valid"] / 
                              gps_processor.validation_stats["total"] 
                              if gps_processor.validation_stats["total"] > 0 else 0),
            "validation_level": gps_processor.validator.config.level.value
        }
    },
    "results": results
}

with open(output_file, 'w') as f:
    json.dump(enhanced_results, f, indent=2, default=str)

logger.info(f"💾 Enhanced results saved to: {output_file}")
```

if **name** == “**main**”:
main()