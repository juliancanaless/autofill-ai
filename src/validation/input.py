"""Input validation for the autofill pipeline."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from ..core.exceptions import ValidationError, DataQualityError
from ..config.settings import get_config
from ..monitoring.logging import get_logger


logger = get_logger(__name__)


class InputValidator:
    """Validates input data for the autofill pipeline."""
    
    def __init__(self):
        try:
            self.config = get_config().validation
        except Exception:
            # Fallback configuration if not initialized
            logger.warning("Using fallback validation configuration")
            from ..config.settings import ValidationConfig
            self.config = ValidationConfig()
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate that the file path exists and has correct extension."""
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        if path.suffix.lower() not in self.config.allowed_file_types:
            raise ValidationError(
                f"File type {path.suffix} not supported. "
                f"Allowed types: {self.config.allowed_file_types}"
            )
        
        logger.debug("File path validation passed", file_path=str(path))
        return path
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame structure and content."""
        if df is None:
            raise ValidationError("DataFrame cannot be None")
        
        if df.empty:
            raise ValidationError("DataFrame cannot be empty")
        
        # Check size limits
        if len(df) > self.config.max_rows:
            raise ValidationError(
                f"DataFrame has {len(df)} rows, maximum allowed is {self.config.max_rows}"
            )
        
        if len(df.columns) > self.config.max_columns:
            raise ValidationError(
                f"DataFrame has {len(df.columns)} columns, maximum allowed is {self.config.max_columns}"
            )
        
        # Check for completely empty columns
        completely_empty_cols = df.columns[df.isnull().all()].tolist()
        if completely_empty_cols:
            raise DataQualityError(
                f"Columns are completely empty: {completely_empty_cols}"
            )
        
        # Check missing data percentage
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > self.config.max_missing_percentage:
            raise DataQualityError(
                f"Dataset has {missing_percentage:.1f}% missing data, "
                f"maximum allowed is {self.config.max_missing_percentage}%"
            )
        
        # Check that columns with missing data have minimum non-null values
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            if non_null_count > 0 and non_null_count < self.config.min_non_null_values:
                raise DataQualityError(
                    f"Column '{column}' has only {non_null_count} non-null values, "
                    f"minimum required is {self.config.min_non_null_values}"
                )
        
        logger.info(
            "DataFrame validation passed",
            rows=len(df),
            columns=len(df.columns),
            missing_percentage=round(missing_percentage, 2)
        )
    
    def validate_column_names(self, df: pd.DataFrame) -> None:
        """Validate column names for potential issues."""
        issues = []
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            issues.append(f"Duplicate column names: {duplicates}")
        
        # Check for empty or whitespace-only column names
        empty_cols = [col for col in df.columns if not str(col).strip()]
        if empty_cols:
            issues.append("Found empty or whitespace-only column names")
        
        # Check for very long column names
        long_cols = [col for col in df.columns if len(str(col)) > 100]
        if long_cols:
            issues.append(f"Very long column names (>100 chars): {long_cols}")
        
        # Check for problematic characters in column names
        problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for col in df.columns:
            col_str = str(col)
            if any(char in col_str for char in problematic_chars):
                issues.append(f"Column '{col}' contains problematic characters")
                break
        
        if issues:
            logger.warning("Column name issues detected", issues=issues)
            # Don't raise exception for column name issues, just warn
        else:
            logger.debug("Column name validation passed")
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality checks."""
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "issues": [],
            "warnings": [],
            "column_types": df.dtypes.to_dict(),
            "missing_data_summary": {}
        }
        
        # Analyze missing data patterns
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                quality_report["missing_data_summary"][column] = {
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 2)
                }
        
        # Check for columns with all identical values (excluding nulls)
        for column in df.columns:
            non_null_values = df[column].dropna()
            if len(non_null_values) > 1 and non_null_values.nunique() == 1:
                quality_report["warnings"].append(
                    f"Column '{column}' has all identical non-null values"
                )
        
        # Check for potential data type issues
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if numeric-looking strings
                non_null_values = df[column].dropna().astype(str)
                if len(non_null_values) > 0:
                    numeric_looking = non_null_values.str.match(r'^-?\d+\.?\d*$').sum()
                    if numeric_looking / len(non_null_values) > 0.8:
                        quality_report["warnings"].append(
                            f"Column '{column}' appears to contain numeric data but is stored as text"
                        )
        
        # Check for very high cardinality in text columns
        for column in df.columns:
            if df[column].dtype == 'object':
                non_null_values = df[column].dropna()
                if len(non_null_values) > 10:
                    cardinality_ratio = non_null_values.nunique() / len(non_null_values)
                    if cardinality_ratio > 0.95:
                        quality_report["warnings"].append(
                            f"Column '{column}' has very high cardinality ({cardinality_ratio:.2%})"
                        )
        
        logger.info(
            "Data quality check completed",
            issues_count=len(quality_report["issues"]),
            warnings_count=len(quality_report["warnings"])
        )
        
        return quality_report
    
    def validate_input(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Main validation method that handles different input types."""
        logger.info("Starting input validation")
        
        # Handle different input types
        if isinstance(data, (str, Path)):
            file_path = self.validate_file_path(data)
            
            # Load the file based on extension
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    raise ValidationError(f"Unsupported file format: {file_path.suffix}")
                    
                logger.info("File loaded successfully", file_path=str(file_path))
                
            except Exception as e:
                raise ValidationError(f"Failed to load file {file_path}: {str(e)}")
        
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        
        else:
            raise ValidationError(f"Unsupported input type: {type(data)}")
        
        # Validate the DataFrame
        self.validate_dataframe(df)
        self.validate_column_names(df)
        
        # Perform quality checks (non-blocking)
        quality_report = self.check_data_quality(df)
        
        # Log summary
        logger.info(
            "Input validation completed successfully",
            shape=df.shape,
            missing_columns=len(quality_report["missing_data_summary"]),
            warnings=len(quality_report["warnings"])
        )
        
        return df


def validate_input(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Convenience function for input validation."""
    validator = InputValidator()
    return validator.validate_input(data)