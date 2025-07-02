"""Output validation for the autofill pipeline."""

import pandas as pd
from typing import Dict, List, Optional, Any
import numpy as np

from ..core.exceptions import ValidationError, DataQualityError
from ..core.types import ImputationResult, QualityAssessment
from ..config.settings import get_config
from ..monitoring.logging import get_logger


logger = get_logger(__name__)


class OutputValidator:
    """Validates output data from the autofill pipeline."""
    
    def __init__(self):
        try:
            self.config = get_config().validation
        except Exception:
            logger.warning("Using fallback validation configuration")
            from ..config.settings import ValidationConfig
            self.config = ValidationConfig()
    
    def validate_filled_dataframe(self, original_df: pd.DataFrame, 
                                filled_df: pd.DataFrame) -> None:
        """Validate the filled DataFrame against the original."""
        
        # Basic structure checks
        if filled_df.shape != original_df.shape:
            raise ValidationError(
                f"Shape mismatch: original {original_df.shape}, filled {filled_df.shape}"
            )
        
        if not filled_df.columns.equals(original_df.columns):
            raise ValidationError("Column names do not match between original and filled DataFrames")
        
        # Check that no new missing values were introduced
        for column in filled_df.columns:
            original_missing = original_df[column].isnull().sum()
            filled_missing = filled_df[column].isnull().sum()
            
            if filled_missing > original_missing:
                raise DataQualityError(
                    f"Column '{column}' has more missing values after imputation: "
                    f"{original_missing} -> {filled_missing}"
                )
        
        # Check that existing non-null values weren't modified
        for column in filled_df.columns:
            original_non_null = original_df[column].notna()
            if original_non_null.any():
                original_values = original_df.loc[original_non_null, column]
                filled_values = filled_df.loc[original_non_null, column]
                
                # For numeric columns, allow small floating-point differences
                if pd.api.types.is_numeric_dtype(original_values) and pd.api.types.is_numeric_dtype(filled_values):
                    if not np.allclose(original_values, filled_values, equal_nan=True, rtol=1e-10):
                        raise DataQualityError(
                            f"Existing non-null values in column '{column}' were modified"
                        )
                else:
                    if not original_values.equals(filled_values):
                        # Check if only string representation differs
                        if not (original_values.astype(str).equals(filled_values.astype(str))):
                            raise DataQualityError(
                                f"Existing non-null values in column '{column}' were modified"
                            )
        
        logger.info(
            "Filled DataFrame validation passed",
            original_shape=original_df.shape,
            filled_shape=filled_df.shape
        )
    
    def validate_imputation_results(self, results: List[ImputationResult], 
                                  original_df: pd.DataFrame) -> None:
        """Validate the imputation results."""
        
        if not isinstance(results, list):
            raise ValidationError("Results must be a list")
        
        if not results:
            logger.info("No imputation results to validate (no missing data)")
            return
        
        # Validate each result
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValidationError(f"Result {i} is not a dictionary")
            
            required_fields = ["index", "column", "original_value", "suggested_value", 
                             "confidence", "method", "reasoning", "timestamp"]
            
            for field in required_fields:
                if field not in result:
                    raise ValidationError(f"Result {i} missing required field: {field}")
            
            # Validate index is within DataFrame bounds
            if not (0 <= result["index"] < len(original_df)):
                raise ValidationError(
                    f"Result {i} has invalid index {result['index']} for DataFrame of length {len(original_df)}"
                )
            
            # Validate column exists
            if result["column"] not in original_df.columns:
                raise ValidationError(
                    f"Result {i} references non-existent column: {result['column']}"
                )
            
            # Validate confidence is between 0 and 1
            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValidationError(
                    f"Result {i} has invalid confidence value: {confidence}"
                )
            
            # Validate that the original location was actually missing
            original_value = original_df.loc[result["index"], result["column"]]
            if pd.notna(original_value):
                raise ValidationError(
                    f"Result {i} claims to fill non-missing value at index {result['index']}, column {result['column']}"
                )
        
        # Check for duplicate results (same index + column)
        seen_locations = set()
        for i, result in enumerate(results):
            location = (result["index"], result["column"])
            if location in seen_locations:
                raise ValidationError(f"Duplicate imputation result for index {result['index']}, column {result['column']}")
            seen_locations.add(location)
        
        logger.info(
            "Imputation results validation passed",
            results_count=len(results)
        )
    
    def validate_quality_assessment(self, assessment: QualityAssessment) -> None:
        """Validate quality assessment structure and values."""
        
        required_fields = ["overall_quality_score", "high_confidence_percentage", 
                          "low_confidence_count", "error_count", "status"]
        
        if isinstance(assessment, dict):
            for field in required_fields:
                if field not in assessment:
                    raise ValidationError(f"Quality assessment missing required field: {field}")
        else:
            # Handle dataclass
            for field in required_fields:
                if not hasattr(assessment, field):
                    raise ValidationError(f"Quality assessment missing required field: {field}")
        
        # Validate score ranges
        score = assessment["overall_quality_score"] if isinstance(assessment, dict) else assessment.overall_quality_score
        if not (0 <= score <= 100):
            raise ValidationError(f"Quality score must be between 0 and 100, got {score}")
        
        high_conf_pct = assessment["high_confidence_percentage"] if isinstance(assessment, dict) else assessment.high_confidence_percentage
        if not (0 <= high_conf_pct <= 100):
            raise ValidationError(f"High confidence percentage must be between 0 and 100, got {high_conf_pct}")
        
        # Validate counts are non-negative
        low_conf_count = assessment["low_confidence_count"] if isinstance(assessment, dict) else assessment.low_confidence_count
        error_count = assessment["error_count"] if isinstance(assessment, dict) else assessment.error_count
        
        if low_conf_count < 0:
            raise ValidationError(f"Low confidence count cannot be negative: {low_conf_count}")
        
        if error_count < 0:
            raise ValidationError(f"Error count cannot be negative: {error_count}")
        
        # Validate status
        valid_statuses = ["excellent", "good", "needs_review", "poor"]
        status = assessment["status"] if isinstance(assessment, dict) else assessment.status
        if status not in valid_statuses:
            raise ValidationError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        
        logger.debug("Quality assessment validation passed")
    
    def check_output_consistency(self, original_df: pd.DataFrame, 
                               filled_df: pd.DataFrame, 
                               results: List[ImputationResult]) -> Dict:
        """Check consistency between filled DataFrame and imputation results."""
        
        consistency_report = {
            "consistent": True,
            "issues": [],
            "filled_count": 0,
            "expected_count": len(results)
        }
        
        # Check that all imputation results match the filled DataFrame
        for result in results:
            idx = result["index"]
            col = result["column"]
            expected_value = result["suggested_value"]
            actual_value = filled_df.loc[idx, col]
            
            # Handle different data types and NaN values
            if pd.isna(expected_value) and pd.isna(actual_value):
                continue  # Both NaN, consistent
            elif pd.isna(expected_value) or pd.isna(actual_value):
                consistency_report["consistent"] = False
                consistency_report["issues"].append(
                    f"NaN mismatch at index {idx}, column {col}: expected {expected_value}, got {actual_value}"
                )
            elif str(expected_value) != str(actual_value):
                # Convert to string for comparison to handle type differences
                consistency_report["consistent"] = False
                consistency_report["issues"].append(
                    f"Value mismatch at index {idx}, column {col}: expected {expected_value}, got {actual_value}"
                )
            else:
                consistency_report["filled_count"] += 1
        
        # Check that no unexpected changes occurred
        for column in original_df.columns:
            original_missing_mask = original_df[column].isnull()
            filled_missing_mask = filled_df[column].isnull()
            
            # Count actual improvements
            actually_filled = (original_missing_mask & ~filled_missing_mask).sum()
            expected_filled = len([r for r in results if r["column"] == column])
            
            if actually_filled != expected_filled:
                consistency_report["consistent"] = False
                consistency_report["issues"].append(
                    f"Column {column}: expected {expected_filled} fills, got {actually_filled}"
                )
        
        logger.info(
            "Output consistency check completed",
            consistent=consistency_report["consistent"],
            issues_count=len(consistency_report["issues"])
        )
        
        return consistency_report
    
    def validate_output(self, original_df: pd.DataFrame, filled_df: pd.DataFrame, 
                       results: List[ImputationResult], 
                       quality_assessment: Optional[QualityAssessment] = None) -> Dict:
        """Main validation method for all outputs."""
        
        logger.info("Starting output validation")
        
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "consistency_report": None
        }
        
        try:
            # Validate filled DataFrame
            self.validate_filled_dataframe(original_df, filled_df)
            
            # Validate imputation results
            self.validate_imputation_results(results, original_df)
            
            # Validate quality assessment if provided
            if quality_assessment is not None:
                self.validate_quality_assessment(quality_assessment)
            
            # Check consistency
            consistency_report = self.check_output_consistency(original_df, filled_df, results)
            validation_report["consistency_report"] = consistency_report
            
            if not consistency_report["consistent"]:
                validation_report["warnings"].extend(consistency_report["issues"])
            
            logger.info("Output validation completed successfully")
            
        except (ValidationError, DataQualityError) as e:
            validation_report["valid"] = False
            validation_report["errors"].append(str(e))
            logger.error("Output validation failed", error=str(e))
            raise
        
        except Exception as e:
            validation_report["valid"] = False
            validation_report["errors"].append(f"Unexpected validation error: {str(e)}")
            logger.error("Unexpected error during output validation", error=str(e), exc_info=True)
            raise ValidationError(f"Output validation failed: {str(e)}")
        
        return validation_report


def validate_output(original_df: pd.DataFrame, filled_df: pd.DataFrame, 
                   results: List[ImputationResult], 
                   quality_assessment: Optional[QualityAssessment] = None) -> Dict:
    """Convenience function for output validation."""
    validator = OutputValidator()
    return validator.validate_output(original_df, filled_df, results, quality_assessment)