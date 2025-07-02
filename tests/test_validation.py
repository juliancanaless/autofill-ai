"""Tests for input and output validation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Add src to path if not already there
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from validation.input import InputValidator, validate_input
    from validation.output import OutputValidator, validate_output
    from core.exceptions import ValidationError, DataQualityError
    from core.types import ImputationResult, QualityAssessment
except ImportError:
    # Fallback import method
    from src.validation.input import InputValidator, validate_input
    from src.validation.output import OutputValidator, validate_output
    from src.core.exceptions import ValidationError, DataQualityError
    from src.core.types import ImputationResult, QualityAssessment


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        self.validator = InputValidator()
    
    def test_validate_file_path_exists(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            result = self.validator.validate_file_path(tmp_path)
            assert result == tmp_path
        finally:
            tmp_path.unlink()
    
    def test_validate_file_path_not_exists(self):
        non_existent = Path("/non/existent/file.csv")
        with pytest.raises(ValidationError, match="File does not exist"):
            self.validator.validate_file_path(non_existent)
    
    def test_validate_file_path_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValidationError, match="File type .txt not supported"):
                self.validator.validate_file_path(tmp_path)
        finally:
            tmp_path.unlink()
    
    def test_validate_dataframe_valid(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, None],
            'B': ['x', 'y', None, 'z']
        })
        self.validator.validate_dataframe(df)  # Should not raise
    
    def test_validate_dataframe_none(self):
        with pytest.raises(ValidationError, match="DataFrame cannot be None"):
            self.validator.validate_dataframe(None)
    
    def test_validate_dataframe_empty(self):
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="DataFrame cannot be empty"):
            self.validator.validate_dataframe(df)
    
    def test_validate_dataframe_too_many_rows(self):
        # Create a large DataFrame
        large_df = pd.DataFrame({'A': range(200000)})  # Exceeds default max_rows
        
        # Mock the config to have a lower limit
        with patch.object(self.validator.config, 'max_rows', 1000):
            with pytest.raises(ValidationError, match="maximum allowed is 1000"):
                self.validator.validate_dataframe(large_df)
    
    def test_validate_dataframe_too_many_columns(self):
        # Create a wide DataFrame
        data = {f'col_{i}': [1, 2] for i in range(1500)}
        wide_df = pd.DataFrame(data)
        
        with patch.object(self.validator.config, 'max_columns', 1000):
            with pytest.raises(ValidationError, match="maximum allowed is 1000"):
                self.validator.validate_dataframe(wide_df)
    
    def test_validate_dataframe_completely_empty_columns(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [None, None, None],  # Completely empty
            'C': ['x', 'y', 'z']
        })
        with pytest.raises(DataQualityError, match="Columns are completely empty"):
            self.validator.validate_dataframe(df)
    
    def test_validate_dataframe_too_much_missing_data(self):
        df = pd.DataFrame({
            'A': [None, None, None, 1],
            'B': [None, None, None, 'x']  # Not completely empty
        })
        with patch.object(self.validator.config, 'max_missing_percentage', 50.0):
            with pytest.raises(DataQualityError, match=r"Dataset has 75\.0% missing data"):
                self.validator.validate_dataframe(df)
    
    def test_validate_dataframe_insufficient_non_null_values(self):
        df = pd.DataFrame({
            'A': [1, None, None, None],  # Only 1 non-null value
            'B': ['x', 'y', 'z', None]
        })
        with patch.object(self.validator.config, 'min_non_null_values', 2):
            with pytest.raises(DataQualityError, match="has only 1 non-null values"):
                self.validator.validate_dataframe(df)
    
    def test_validate_column_names_duplicates(self):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'A'])
        # Should warn but not raise exception
        self.validator.validate_column_names(df)
    
    def test_check_data_quality(self):
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', 'x', 'x', 'x'],  # All identical
            'C': ['1', '2', '3', '4']   # Numeric-looking strings
        })
        
        quality_report = self.validator.check_data_quality(df)
        
        assert quality_report['total_rows'] == 4
        assert quality_report['total_columns'] == 3
        assert 'A' in quality_report['missing_data_summary']
        assert len(quality_report['warnings']) > 0
    
    def test_validate_input_csv_file(self):
        # Create a temporary CSV file
        df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', None, 'z']})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = Path(tmp.name)
        
        try:
            result_df = self.validator.validate_input(tmp_path)
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 3
            assert len(result_df.columns) == 2
        finally:
            tmp_path.unlink()
    
    def test_validate_input_dataframe(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result_df = self.validator.validate_input(df)
        
        assert isinstance(result_df, pd.DataFrame)
        pd.testing.assert_frame_equal(result_df, df)
    
    def test_validate_input_invalid_type(self):
        with pytest.raises(ValidationError, match="Unsupported input type"):
            self.validator.validate_input(123)


class TestOutputValidator:
    """Test output validation functionality."""
    
    def setup_method(self):
        self.validator = OutputValidator()
        
        # Create test data
        self.original_df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', None, 'z', 'w'],
            'C': [10.5, 20.0, 30.5, None]
        })
        
        self.filled_df = pd.DataFrame({
            'A': [1, 2, 3, 4],  # Filled missing value
            'B': ['x', 'y', 'z', 'w'],  # Filled missing value
            'C': [10.5, 20.0, 30.5, 25.0]  # Filled missing value
        })
        
        self.results = [
            {
                "index": 2,
                "column": "A",
                "original_value": None,
                "suggested_value": 3,
                "confidence": 0.8,
                "method": "knn_imputation",
                "reasoning": "KNN prediction",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "index": 1,
                "column": "B",
                "original_value": None,
                "suggested_value": "y",
                "confidence": 0.7,
                "method": "llm_categorical",
                "reasoning": "LLM prediction",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "index": 3,
                "column": "C",
                "original_value": None,
                "suggested_value": 25.0,
                "confidence": 0.9,
                "method": "statistical",
                "reasoning": "Statistical imputation",
                "timestamp": "2024-01-01T00:00:00"
            }
        ]
    
    def test_validate_filled_dataframe_valid(self):
        self.validator.validate_filled_dataframe(self.original_df, self.filled_df)
        # Should not raise exception
    
    def test_validate_filled_dataframe_shape_mismatch(self):
        wrong_shape_df = self.filled_df.iloc[:2]  # Different number of rows
        
        with pytest.raises(ValidationError, match="Shape mismatch"):
            self.validator.validate_filled_dataframe(self.original_df, wrong_shape_df)
    
    def test_validate_filled_dataframe_column_mismatch(self):
        wrong_cols_df = self.filled_df.rename(columns={'A': 'X'})
        
        with pytest.raises(ValidationError, match="Column names do not match"):
            self.validator.validate_filled_dataframe(self.original_df, wrong_cols_df)
    
    def test_validate_filled_dataframe_new_missing_values(self):
        # Create DataFrame with new missing values by not filling a missing value that should have been filled
        # Original B: ['x', None, 'z', 'w'] - 1 missing value at index 1  
        # Filled B: ['x', 'y', 'z', 'w'] - 0 missing values
        # Bad B: ['x', None, 'z', 'w'] - 1 missing value (didn't fill the original missing)
        
        # But we need MORE missing values, so let's not fill ALL missing values
        bad_filled_df = self.original_df.copy()  # Start with original (unfilled) data
        # Only fill some columns, leave others unfilled to create "new" missing values relative to filled_df
        bad_filled_df.loc[2, 'A'] = 3  # Fill this one
        # Don't fill B[1] and don't fill C[3] - so we have same missing as original
        
        # This doesn't work either. Let me try a different approach.
        # Create a completely new test case that's simpler
        simple_original = pd.DataFrame({'X': [1, 2, 3]})  # No missing values
        simple_filled = pd.DataFrame({'X': [1, None, 3]})  # Introduced missing value
        
        with pytest.raises(DataQualityError, match=r"Column 'X' has more missing values after imputation: 0 -> 1"):
            self.validator.validate_filled_dataframe(simple_original, simple_filled)
    
    def test_validate_filled_dataframe_modified_existing_values(self):
        # Modify an existing non-null value
        bad_filled_df = self.filled_df.copy()
        bad_filled_df.loc[0, 'A'] = 999  # Change existing value
        
        with pytest.raises(DataQualityError, match="Existing non-null values.*were modified"):
            self.validator.validate_filled_dataframe(self.original_df, bad_filled_df)
    
    def test_validate_imputation_results_valid(self):
        self.validator.validate_imputation_results(self.results, self.original_df)
        # Should not raise exception
    
    def test_validate_imputation_results_not_list(self):
        with pytest.raises(ValidationError, match="Results must be a list"):
            self.validator.validate_imputation_results("not a list", self.original_df)
    
    def test_validate_imputation_results_empty(self):
        # Empty results should be valid (no missing data case)
        self.validator.validate_imputation_results([], self.original_df)
    
    def test_validate_imputation_results_missing_field(self):
        bad_result = self.results[0].copy()
        del bad_result["confidence"]  # Remove required field
        
        with pytest.raises(ValidationError, match="missing required field: confidence"):
            self.validator.validate_imputation_results([bad_result], self.original_df)
    
    def test_validate_imputation_results_invalid_index(self):
        bad_result = self.results[0].copy()
        bad_result["index"] = 999  # Invalid index
        
        with pytest.raises(ValidationError, match="has invalid index 999"):
            self.validator.validate_imputation_results([bad_result], self.original_df)
    
    def test_validate_imputation_results_nonexistent_column(self):
        bad_result = self.results[0].copy()
        bad_result["column"] = "NonExistent"
        
        with pytest.raises(ValidationError, match="references non-existent column"):
            self.validator.validate_imputation_results([bad_result], self.original_df)
    
    def test_validate_imputation_results_invalid_confidence(self):
        bad_result = self.results[0].copy()
        bad_result["confidence"] = 1.5  # Invalid confidence > 1
        
        with pytest.raises(ValidationError, match="has invalid confidence value"):
            self.validator.validate_imputation_results([bad_result], self.original_df)
    
    def test_validate_imputation_results_non_missing_original(self):
        bad_result = self.results[0].copy()
        bad_result["index"] = 0  # Index 0 has non-missing value in column A
        
        with pytest.raises(ValidationError, match="claims to fill non-missing value"):
            self.validator.validate_imputation_results([bad_result], self.original_df)
    
    def test_validate_imputation_results_duplicates(self):
        duplicate_result = self.results[0].copy()
        results_with_duplicate = self.results + [duplicate_result]
        
        with pytest.raises(ValidationError, match="Duplicate imputation result"):
            self.validator.validate_imputation_results(results_with_duplicate, self.original_df)
    
    def test_validate_quality_assessment_valid(self):
        assessment = {
            "overall_quality_score": 85.0,
            "high_confidence_percentage": 75.0,
            "low_confidence_count": 2,
            "error_count": 0,
            "status": "good"
        }
        self.validator.validate_quality_assessment(assessment)
        # Should not raise exception
    
    def test_validate_quality_assessment_missing_field(self):
        assessment = {
            "overall_quality_score": 85.0,
            # Missing other required fields
        }
        with pytest.raises(ValidationError, match="missing required field"):
            self.validator.validate_quality_assessment(assessment)
    
    def test_validate_quality_assessment_invalid_score(self):
        assessment = {
            "overall_quality_score": 150.0,  # Invalid score > 100
            "high_confidence_percentage": 75.0,
            "low_confidence_count": 2,
            "error_count": 0,
            "status": "good"
        }
        with pytest.raises(ValidationError, match="Quality score must be between 0 and 100"):
            self.validator.validate_quality_assessment(assessment)
    
    def test_validate_quality_assessment_invalid_status(self):
        assessment = {
            "overall_quality_score": 85.0,
            "high_confidence_percentage": 75.0,
            "low_confidence_count": 2,
            "error_count": 0,
            "status": "invalid_status"
        }
        with pytest.raises(ValidationError, match="Invalid status"):
            self.validator.validate_quality_assessment(assessment)
    
    def test_check_output_consistency_consistent(self):
        consistency_report = self.validator.check_output_consistency(
            self.original_df, self.filled_df, self.results
        )
        
        assert consistency_report["consistent"] is True
        assert len(consistency_report["issues"]) == 0
        assert consistency_report["filled_count"] == 3
    
    def test_check_output_consistency_value_mismatch(self):
        # Create filled DataFrame that doesn't match results
        inconsistent_filled_df = self.filled_df.copy()
        inconsistent_filled_df.loc[2, 'A'] = 999  # Different from result
        
        consistency_report = self.validator.check_output_consistency(
            self.original_df, inconsistent_filled_df, self.results
        )
        
        assert consistency_report["consistent"] is False
        assert len(consistency_report["issues"]) > 0
    
    def test_validate_output_complete(self):
        validation_report = self.validator.validate_output(
            self.original_df, self.filled_df, self.results
        )
        
        assert validation_report["valid"] is True
        assert len(validation_report["errors"]) == 0
        assert validation_report["consistency_report"]["consistent"] is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_validate_input_function(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result = validate_input(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_validate_output_function(self):
        original_df = pd.DataFrame({'A': [1, None, 3]})
        filled_df = pd.DataFrame({'A': [1, 2, 3]})
        results = [{
            "index": 1,
            "column": "A",
            "original_value": None,
            "suggested_value": 2,
            "confidence": 0.8,
            "method": "test",
            "reasoning": "test",
            "timestamp": "2024-01-01T00:00:00"
        }]
        
        validation_report = validate_output(original_df, filled_df, results)
        assert validation_report["valid"] is True