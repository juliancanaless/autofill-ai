"""
Production-ready wrapper for the AutoFill Pipeline
Adds enterprise features while maintaining compatibility with the original implementation
"""

import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .autofill_pipeline import AutoFillPipeline
from .config.settings import init_config, get_config
from .validation.input import validate_input
from .validation.output import validate_output
from .monitoring.logging import get_logger
from .monitoring.metrics import track_performance, get_metrics_collector
from .llm.cache import get_llm_cache
from .core.exceptions import ValidationError, ConfigurationError


logger = get_logger(__name__)


class ProductionAutoFillPipeline:
    """
    Production-ready wrapper around the original AutoFillPipeline.
    Adds validation, monitoring, caching, and error handling.
    """
    
    def __init__(self, llm_provider: str = "groq", 
                 config_path: Optional[Path] = None,
                 enable_validation: bool = True,
                 enable_monitoring: bool = True,
                 enable_caching: bool = True):
        """
        Initialize the production pipeline.
        
        Args:
            llm_provider: LLM provider to use ("groq", "openai", "ollama")
            config_path: Path to configuration file (optional)
            enable_validation: Enable input/output validation
            enable_monitoring: Enable performance monitoring
            enable_caching: Enable LLM response caching
        """
        self.llm_provider = llm_provider
        self.enable_validation = enable_validation
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        
        # Initialize configuration
        try:
            # Try to get existing config
            self.config = get_config()
            logger.info("Using existing configuration")
        except ConfigurationError:
            # Initialize new config
            self.config = init_config(llm_provider)
            logger.info("Initialized new configuration", provider=llm_provider)
        
        # Initialize the underlying pipeline
        self.pipeline = AutoFillPipeline(
            llm_provider=llm_provider,
            model_name=self.config.llm.model_name,
            api_key=self.config.llm.api_key
        )
        
        # Initialize components
        self.metrics_collector = get_metrics_collector() if enable_monitoring else None
        self.cache = get_llm_cache() if enable_caching else None
        
        logger.info(
            "Production pipeline initialized",
            provider=llm_provider,
            validation=enable_validation,
            monitoring=enable_monitoring,
            caching=enable_caching
        )
    
    def analyze_with_validation(self, data: Union[str, Path, pd.DataFrame]) -> Dict:
        """
        Analyze dataset with input validation and monitoring.
        
        Args:
            data: Input data (file path or DataFrame)
            
        Returns:
            Analysis results with validation information
        """
        with track_performance("dataset_analysis_with_validation"):
            # Input validation
            if self.enable_validation:
                logger.info("Performing input validation")
                validated_df = validate_input(data)
            else:
                if isinstance(data, (str, Path)):
                    validated_df = pd.read_csv(data)
                else:
                    validated_df = data.copy()
            
            # Perform analysis using original pipeline
            logger.info("Starting dataset analysis", shape=validated_df.shape)
            analysis = self.pipeline.analyze_dataset(validated_df)
            
            # Add validation metadata
            if self.enable_validation:
                analysis["validation"] = {
                    "input_validated": True,
                    "validation_timestamp": time.time()
                }
            
            logger.info(
                "Dataset analysis completed",
                complexity_score=analysis.get("complexity_score", 0),
                missing_columns=len(analysis.get("missing_data_summary", {}))
            )
            
            return analysis
    
    def fill_with_monitoring(self, df: pd.DataFrame, 
                           validate_output_flag: bool = None) -> Tuple[pd.DataFrame, List[Dict], Dict]:
        """
        Fill missing data with comprehensive monitoring and validation.
        
        Args:
            df: Input DataFrame
            validate_output_flag: Override output validation setting
            
        Returns:
            Tuple of (filled_df, results, comprehensive_report)
        """
        validate_output_flag = validate_output_flag if validate_output_flag is not None else self.enable_validation
        
        with track_performance("data_imputation_with_monitoring") as tracker:
            start_time = time.time()
            
            # Store original for validation
            original_df = df.copy()
            
            # Perform imputation using original pipeline
            logger.info("Starting data imputation", shape=df.shape)
            filled_df, results = self.pipeline.fill_missing_data(df)
            
            # Generate completion report
            completion_report = self.pipeline.generate_completion_report(original_df, filled_df, results)
            
            # Output validation
            validation_report = None
            if validate_output_flag:
                logger.info("Performing output validation")
                try:
                    validation_report = validate_output(original_df, filled_df, results)
                    logger.info("Output validation passed")
                except Exception as e:
                    logger.error("Output validation failed", error=str(e))
                    validation_report = {"valid": False, "errors": [str(e)]}
            
            # Record metrics
            if self.enable_monitoring and results:
                processing_time = time.time() - start_time
                
                # Record imputation metrics by method
                method_stats = {}
                for result in results:
                    method = result["method"]
                    if method not in method_stats:
                        method_stats[method] = {"count": 0, "confidence_sum": 0}
                    method_stats[method]["count"] += 1
                    method_stats[method]["confidence_sum"] += result["confidence"]
                
                for method, stats in method_stats.items():
                    avg_confidence = stats["confidence_sum"] / stats["count"]
                    self.metrics_collector.record_imputation(
                        column="aggregated",
                        method=method,
                        count=stats["count"],
                        average_confidence=avg_confidence,
                        duration_seconds=processing_time
                    )
            
            # Create comprehensive report
            comprehensive_report = {
                "completion_report": completion_report,
                "validation_report": validation_report,
                "performance": {
                    "processing_time_seconds": time.time() - start_time,
                    "rows_processed": len(df),
                    "values_filled": len(results)
                },
                "cache_stats": self.cache.get_stats() if self.cache else None
            }
            
            logger.info(
                "Data imputation completed",
                processing_time=comprehensive_report["performance"]["processing_time_seconds"],
                values_filled=len(results),
                validation_passed=validation_report["valid"] if validation_report else None
            )
            
            return filled_df, results, comprehensive_report
    
    def process_file(self, file_path: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Process a file end-to-end with full production features.
        
        Args:
            file_path: Input file path
            output_path: Output file path (optional)
            
        Returns:
            Processing summary report
        """
        file_path = Path(file_path)
        
        with track_performance("file_processing_end_to_end"):
            logger.info("Starting file processing", file_path=str(file_path))
            
            # Analyze dataset
            analysis = self.analyze_with_validation(file_path)
            
            # Load validated data
            if self.enable_validation:
                df = validate_input(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Fill missing data
            filled_df, results, comprehensive_report = self.fill_with_monitoring(df)
            
            # Save output if path provided
            if output_path:
                output_path = Path(output_path)
                filled_df.to_csv(output_path, index=False)
                logger.info("Output saved", output_path=str(output_path))
                comprehensive_report["output_path"] = str(output_path)
            
            # Create summary report
            summary_report = {
                "input_file": str(file_path),
                "analysis": analysis,
                "processing_report": comprehensive_report,
                "summary": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "missing_values_filled": len(results),
                    "processing_time": comprehensive_report["performance"]["processing_time_seconds"],
                    "quality_score": comprehensive_report["completion_report"]["quality_assessment"]["overall_quality_score"]
                }
            }
            
            logger.info(
                "File processing completed",
                input_file=str(file_path),
                rows=len(df),
                filled_values=len(results),
                quality_score=summary_report["summary"]["quality_score"]
            )
            
            return summary_report
    
    def batch_process(self, file_paths: List[Union[str, Path]], 
                     output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Process multiple files in batch with aggregated reporting.
        
        Args:
            file_paths: List of input file paths
            output_dir: Output directory for processed files
            
        Returns:
            List of processing reports for each file
        """
        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        reports = []
        total_start_time = time.time()
        
        logger.info("Starting batch processing", file_count=len(file_paths))
        
        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            
            try:
                # Determine output path
                if output_dir:
                    output_path = output_dir / f"filled_{file_path.name}"
                else:
                    output_path = None
                
                # Process file
                report = self.process_file(file_path, output_path)
                report["batch_index"] = i
                reports.append(report)
                
                logger.info(
                    "Batch file completed",
                    file=str(file_path),
                    progress=f"{i+1}/{len(file_paths)}"
                )
                
            except Exception as e:
                error_report = {
                    "input_file": str(file_path),
                    "batch_index": i,
                    "error": str(e),
                    "success": False
                }
                reports.append(error_report)
                
                logger.error(
                    "Batch file failed",
                    file=str(file_path),
                    error=str(e),
                    progress=f"{i+1}/{len(file_paths)}"
                )
        
        # Generate batch summary
        total_time = time.time() - total_start_time
        successful_files = [r for r in reports if "error" not in r]
        failed_files = [r for r in reports if "error" in r]
        
        batch_summary = {
            "total_files": len(file_paths),
            "successful_files": len(successful_files),
            "failed_files": len(failed_files),
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(file_paths) if file_paths else 0
        }
        
        logger.info(
            "Batch processing completed",
            **batch_summary
        )
        
        # Add summary to all reports
        for report in reports:
            report["batch_summary"] = batch_summary
        
        return reports
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get comprehensive metrics summary."""
        if not self.enable_monitoring:
            return {"monitoring_disabled": True}
        
        collector = get_metrics_collector()
        
        return {
            "performance": collector.get_performance_summary(hours),
            "api_usage": collector.get_api_usage_summary(hours),
            "imputation": collector.get_imputation_summary(hours),
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def get_health_status(self) -> Dict:
        """Get health status for monitoring systems."""
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check pipeline health
        try:
            test_df = pd.DataFrame({"test": [1, None]})
            self.pipeline.analyze_dataset(test_df)
            status["components"]["pipeline"] = "healthy"
        except Exception as e:
            status["components"]["pipeline"] = f"unhealthy: {str(e)}"
            status["status"] = "unhealthy"
        
        # Check cache health
        if self.cache:
            try:
                cache_stats = self.cache.get_stats()
                status["components"]["cache"] = "healthy"
                status["cache_size"] = cache_stats.get("size", 0)
            except Exception as e:
                status["components"]["cache"] = f"unhealthy: {str(e)}"
        
        # Check metrics collector health
        if self.metrics_collector:
            try:
                metrics_summary = self.metrics_collector.get_performance_summary(1)
                status["components"]["metrics"] = "healthy"
            except Exception as e:
                status["components"]["metrics"] = f"unhealthy: {str(e)}"
        
        return status
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up production pipeline resources")
        
        if hasattr(self.pipeline, 'cleanup'):
            self.pipeline.cleanup()
        
        if self.cache:
            # Optionally clear cache on cleanup
            pass
        
        logger.info("Production pipeline cleanup completed")


def main():
    """Example usage of the production pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production AutoFill Pipeline")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-p", "--provider", default="groq", choices=["groq", "openai", "ollama"])
    parser.add_argument("--no-validation", action="store_true", help="Disable validation")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--no-caching", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionAutoFillPipeline(
        llm_provider=args.provider,
        enable_validation=not args.no_validation,
        enable_monitoring=not args.no_monitoring,
        enable_caching=not args.no_caching
    )
    
    # Process file
    try:
        report = pipeline.process_file(args.input_file, args.output)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Input file: {report['input_file']}")
        print(f"Rows processed: {report['summary']['total_rows']}")
        print(f"Values filled: {report['summary']['missing_values_filled']}")
        print(f"Quality score: {report['summary']['quality_score']:.1f}/100")
        print(f"Processing time: {report['summary']['processing_time']:.1f}s")
        
        if args.output:
            print(f"Output saved to: {args.output}")
        
        # Show metrics if monitoring enabled
        if not args.no_monitoring:
            metrics = pipeline.get_metrics_summary(1)
            if "performance" in metrics and metrics["performance"]["total_operations"] > 0:
                print(f"\nPerformance metrics:")
                for op, stats in metrics["performance"]["operations"].items():
                    print(f"  {op}: {stats['avg_duration']:.2f}s avg")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    finally:
        pipeline.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())