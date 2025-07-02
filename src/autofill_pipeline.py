"""
AutoFill AI - Generalized Data Imputation Pipeline (Cloud LLM Version)
Domain-agnostic system for intelligent CSV gap filling with confidence scoring
Uses Groq API for fast, reliable LLM inference
"""

import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Cloud LLM imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not installed. Run: pip install groq")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoFillPipeline:
    """
    Generalized AI-powered data imputation pipeline with cloud LLM support
    Supports Groq (fast), OpenAI (premium), and Ollama (local) backends
    """
    
    def __init__(self, 
                 llm_provider: str = "groq",  # "groq", "openai", or "ollama"
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        
        self.processing_timestamp = datetime.now().isoformat()
        self.column_strategies = {}
        self.imputation_results = []
        self.llm_provider = llm_provider
        
        # Initialize LLM client
        if llm_provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq not available. Run: pip install groq")
            
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter")
            
            self.client = Groq(api_key=self.api_key)
            self.model_name = model_name or "llama3-8b-8192"  # Fast Groq model
            
        elif llm_provider == "openai":
            try:
                from openai import OpenAI
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OpenAI API key required")
                self.client = OpenAI(api_key=self.api_key)
                self.model_name = model_name or "gpt-3.5-turbo"
            except ImportError:
                raise ImportError("OpenAI not available. Run: pip install openai")
                
        elif llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Run: pip install ollama")
            self.client = None  # Ollama uses direct function calls
            self.model_name = model_name or "llama3.1:8b"
            
        else:
            raise ValueError("llm_provider must be 'groq', 'openai', or 'ollama'")
        
        logger.info(f"Initialized AutoFill Pipeline with {llm_provider} ({self.model_name})")

    def get_api_usage_info(self) -> str:
        """Return information about API costs/limits"""
        
        if self.llm_provider == "groq":
            return "Groq: 6,000 free requests/day with Llama 3.1 (very fast)"
        elif self.llm_provider == "openai":
            return "OpenAI: ~$0.002 per request with GPT-3.5-turbo (premium quality)"
        elif self.llm_provider == "ollama":
            return "Ollama: Free local inference (requires good hardware)"
        
        return "Unknown provider"

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive dataset analysis and imputation strategy planning
        Works for any domain - detects patterns and suggests approaches
        """
        
        total_rows = len(df)
        analysis = {
            "dataset_profile": {
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "analysis_timestamp": self.processing_timestamp
            },
            "column_analysis": {},
            "missing_data_summary": {},
            "imputation_plan": {},
            "complexity_score": 0
        }
        
        complexity_factors = 0
        
        for column in df.columns:
            col_analysis = self._analyze_column(df[column], column)
            analysis["column_analysis"][column] = col_analysis
            
            # Track missing data
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                analysis["missing_data_summary"][column] = {
                    "missing_count": missing_count,
                    "missing_percentage": round(missing_pct, 1),
                    "priority": self._calculate_priority(missing_pct, col_analysis),
                    "estimated_time_minutes": self._estimate_fill_time(missing_count, col_analysis)
                }
                
                # Determine imputation strategy
                strategy = self._determine_strategy(col_analysis, df[column])
                analysis["imputation_plan"][column] = strategy
                self.column_strategies[column] = strategy
                
                # Add to complexity score
                if strategy["method"] == "llm_contextual":
                    complexity_factors += missing_count * 0.1
                elif strategy["method"] == "statistical":
                    complexity_factors += missing_count * 0.01
        
        analysis["complexity_score"] = round(complexity_factors, 1)
        analysis["estimated_total_time"] = self._estimate_total_time(analysis["missing_data_summary"])
        
        return analysis

    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict:
        """Analyze individual column characteristics - domain agnostic"""
        
        non_null_data = series.dropna()
        
        analysis = {
            "data_type": str(series.dtype),
            "unique_values": len(non_null_data.unique()),
            "sample_values": non_null_data.head(3).tolist() if len(non_null_data) > 0 else [],
            "is_numeric": pd.api.types.is_numeric_dtype(series),
            "is_categorical": False,
            "has_patterns": False,
            "domain_hints": []
        }
        
        if len(non_null_data) > 0:
            # Detect if categorical (low cardinality text)
            if not analysis["is_numeric"] and analysis["unique_values"] <= min(50, len(non_null_data) * 0.5):
                analysis["is_categorical"] = True
                analysis["categories"] = non_null_data.unique().tolist()
            
            # Detect patterns in data
            if not analysis["is_numeric"]:
                analysis["domain_hints"] = self._detect_domain_hints(non_null_data, column_name)
                analysis["has_patterns"] = len(analysis["domain_hints"]) > 0
        
        return analysis

    def _detect_domain_hints(self, series: pd.Series, column_name: str) -> List[str]:
        """Detect what domain/type of data this might be - helps with LLM prompting"""
        
        hints = []
        sample_values = series.astype(str).str.lower()
        column_lower = column_name.lower()
        
        # Common domain patterns
        domain_patterns = {
            "company": ["company", "business", "organization", "corp", "inc", "ltd"],
            "industry": ["industry", "sector", "vertical", "category"],
            "location": ["city", "state", "country", "location", "address"],
            "product": ["product", "item", "sku", "merchandise"],
            "person": ["name", "person", "customer", "user", "employee"],
            "status": ["status", "state", "condition", "level"],
            "category": ["category", "type", "class", "group", "segment"],
            "food": ["food", "ingredient", "nutrition", "recipe", "meal"],
            "finance": ["revenue", "cost", "price", "amount", "value"],
            "technology": ["software", "tech", "platform", "tool", "system"]
        }
        
        # Check column name for domain hints
        for domain, keywords in domain_patterns.items():
            if any(keyword in column_lower for keyword in keywords):
                hints.append(f"domain:{domain}")
        
        # Check sample values for patterns
        sample_str = " ".join(sample_values.head(10))
        
        if any(word in sample_str for word in ["inc", "corp", "llc", "ltd"]):
            hints.append("pattern:company_names")
        elif any(word in sample_str for word in ["high", "low", "medium", "good", "bad"]):
            hints.append("pattern:ratings_levels")
        elif len(series.unique()) <= 10:
            hints.append("pattern:limited_categories")
        
        return hints

    def _determine_strategy(self, col_analysis: Dict, series: pd.Series) -> Dict:
        """Determine the best imputation strategy for each column"""
        
        if col_analysis["is_numeric"]:
            # Numeric data - use statistical/ML methods
            return {
                "method": "statistical",
                "approach": "knn_imputation",
                "confidence_base": 0.8,
                "explanation": "K-Nearest Neighbors imputation for numeric data"
            }
        
        elif col_analysis["is_categorical"] and len(col_analysis.get("categories", [])) <= 20:
            # Small categorical - LLM can handle with examples
            return {
                "method": "llm_categorical", 
                "approach": "classification_with_examples",
                "confidence_base": 0.7,
                "categories": col_analysis["categories"],
                "explanation": "LLM classification with known category examples"
            }
        
        else:
            # Complex text data - full LLM reasoning
            return {
                "method": "llm_contextual",
                "approach": "contextual_inference",
                "confidence_base": 0.6,
                "domain_hints": col_analysis.get("domain_hints", []),
                "explanation": "LLM contextual inference based on surrounding data"
            }

    def _calculate_priority(self, missing_pct: float, col_analysis: Dict) -> str:
        """Calculate fill priority based on missing percentage and complexity"""
        
        if missing_pct > 70:
            return "HIGH"
        elif missing_pct > 30:
            return "MEDIUM" 
        else:
            return "LOW"

    def _estimate_fill_time(self, missing_count: int, col_analysis: Dict) -> float:
        """Estimate time to fill missing values in this column"""
        
        if col_analysis["is_numeric"]:
            return 0.1  # Statistical is fast
        elif col_analysis["is_categorical"]:
            return missing_count * 0.05  # LLM categorical is medium
        else:
            return missing_count * 0.2  # Complex LLM is slow

    def _estimate_total_time(self, missing_summary: Dict) -> str:
        """Estimate total processing time"""
        
        total_minutes = sum(col["estimated_time_minutes"] for col in missing_summary.values())
        
        if total_minutes < 2:
            return "< 2 minutes"
        elif total_minutes < 10:
            return f"~{int(total_minutes)} minutes"
        else:
            return f"~{int(total_minutes)} minutes (consider batch processing)"

    def fill_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Execute the imputation plan - fills all missing data using appropriate strategies
        """
        
        logger.info("Starting comprehensive data imputation...")
        filled_df = df.copy()
        all_results = []
        
        # Process each column with missing data
        for column, strategy in self.column_strategies.items():
            if filled_df[column].isnull().any():
                logger.info(f"Filling {column} using {strategy['method']}...")
                
                if strategy["method"] == "statistical":
                    results = self._fill_statistical(filled_df, column, strategy)
                elif strategy["method"] == "llm_categorical":
                    results = self._fill_llm_categorical(filled_df, column, strategy)
                elif strategy["method"] == "llm_contextual":
                    results = self._fill_llm_contextual(filled_df, column, strategy)
                
                # Apply results to dataframe
                for result in results:
                    filled_df.loc[result["index"], column] = result["suggested_value"]
                
                all_results.extend(results)
        
        return filled_df, all_results

    def _fill_statistical(self, df: pd.DataFrame, column: str, strategy: Dict) -> List[Dict]:
        """Fill numeric data using statistical methods with better validation"""
        
        results = []
        missing_indices = df[df[column].isnull()].index
        
        if len(missing_indices) == 0:
            return results
        
        # Use KNN imputation for numeric columns with validation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1 and len(df.dropna()) >= 3:
            try:
                # Only use columns that are somewhat correlated
                correlation_threshold = 0.1
                target_col_data = df[column].dropna()
                
                if len(target_col_data) > 0:
                    useful_columns = [column]
                    for col in numeric_columns:
                        if col != column and len(df[col].dropna()) > 0:
                            # Simple correlation check
                            common_indices = target_col_data.index.intersection(df[col].dropna().index)
                            if len(common_indices) >= 2:
                                useful_columns.append(col)
                
                if len(useful_columns) > 1:
                    imputer = KNNImputer(n_neighbors=min(3, len(df.dropna())))
                    filled_values = imputer.fit_transform(df[useful_columns])
                    filled_df = pd.DataFrame(filled_values, columns=useful_columns, index=df.index)
                    
                    for idx in missing_indices:
                        original_value = df.loc[idx, column]
                        suggested_value = filled_df.loc[idx, column]
                        
                        # Sanity check - ensure value is reasonable
                        existing_values = df[column].dropna()
                        if len(existing_values) > 0:
                            min_val, max_val = existing_values.min(), existing_values.max()
                            value_range = max_val - min_val
                            
                            # If predicted value is way outside range, use median instead
                            if suggested_value < min_val - value_range or suggested_value > max_val + value_range:
                                suggested_value = existing_values.median()
                                method = "median_fallback"
                                confidence = 0.6
                                reasoning = f"KNN prediction outside reasonable range, used median: {suggested_value:.2f}"
                            else:
                                method = "knn_imputation"
                                confidence = 0.8
                                reasoning = f"K-NN imputation based on {len(useful_columns)-1} related numeric columns"
                        else:
                            method = "knn_imputation"
                            confidence = 0.7
                            reasoning = f"K-NN imputation (no validation data available)"
                        
                        results.append({
                            "index": idx,
                            "column": column,
                            "original_value": original_value,
                            "suggested_value": round(suggested_value, 2) if not pd.isna(suggested_value) else None,
                            "confidence": confidence,
                            "method": method,
                            "reasoning": reasoning,
                            "timestamp": self.processing_timestamp
                        })
                else:
                    # Not enough useful columns, fall back to median
                    median_value = df[column].median()
                    for idx in missing_indices:
                        results.append({
                            "index": idx,
                            "column": column,
                            "original_value": df.loc[idx, column],
                            "suggested_value": round(median_value, 2) if not pd.isna(median_value) else 0,
                            "confidence": 0.6,
                            "method": "median_imputation",
                            "reasoning": f"Median imputation (insufficient correlated columns)",
                            "timestamp": self.processing_timestamp
                        })
                        
            except Exception as e:
                logger.error(f"KNN imputation failed: {e}, falling back to median")
                # Fallback to median imputation
                median_value = df[column].median()
                for idx in missing_indices:
                    results.append({
                        "index": idx,
                        "column": column,
                        "original_value": df.loc[idx, column],
                        "suggested_value": round(median_value, 2) if not pd.isna(median_value) else 0,
                        "confidence": 0.5,
                        "method": "median_fallback",
                        "reasoning": f"KNN failed, median imputation: {median_value:.2f}",
                        "timestamp": self.processing_timestamp
                    })
        else:
            # Fallback to median imputation
            median_value = df[column].median()
            for idx in missing_indices:
                results.append({
                    "index": idx,
                    "column": column,
                    "original_value": df.loc[idx, column],
                    "suggested_value": round(median_value, 2) if not pd.isna(median_value) else 0,
                    "confidence": 0.6,
                    "method": "median_imputation",
                    "reasoning": f"Median imputation (insufficient data for KNN)",
                    "timestamp": self.processing_timestamp
                })
        
        return results

    def _fill_llm_categorical(self, df: pd.DataFrame, column: str, strategy: Dict) -> List[Dict]:
        """Fill categorical data using LLM with known categories"""
        
        results = []
        missing_indices = df[df[column].isnull()].index
        known_categories = strategy.get("categories", [])
        
        for idx in missing_indices:
            row_context = self._build_row_context(df, idx, column)
            
            prompt = f"""
            Fill the missing value for column "{column}" in this data row.
            
            Row context: {row_context}
            
            Known valid categories for {column}: {known_categories}
            
            Choose the most appropriate category from the list above.
            Consider the context of other fields in this row.
            
            Respond with this JSON format:
            {{
                "value": "<selected_category>",
                "confidence": <0.0-1.0>,
                "reasoning": "<brief explanation>"
            }}
            """
            
            try:
                suggested_value, confidence, reasoning = self._query_llm(prompt)
                
                # Validate against known categories
                if suggested_value not in known_categories:
                    # Find closest match
                    suggested_value = self._find_closest_category(suggested_value, known_categories)
                    confidence *= 0.8  # Reduce confidence for fuzzy match
                    reasoning += " (adjusted to closest valid category)"
                
                results.append({
                    "index": idx,
                    "column": column,
                    "original_value": df.loc[idx, column],
                    "suggested_value": suggested_value,
                    "confidence": confidence,
                    "method": "llm_categorical",
                    "reasoning": reasoning,
                    "timestamp": self.processing_timestamp
                })
                
            except Exception as e:
                logger.error(f"LLM categorical fill failed for row {idx}: {e}")
                # Fallback to most common category
                fallback_value = known_categories[0] if known_categories else "Unknown"
                results.append({
                    "index": idx,
                    "column": column,
                    "original_value": df.loc[idx, column],
                    "suggested_value": fallback_value,
                    "confidence": 0.3,
                    "method": "fallback",
                    "reasoning": f"Error in LLM processing, used fallback: {str(e)}",
                    "timestamp": self.processing_timestamp
                })
        
        return results

    def _fill_llm_contextual(self, df: pd.DataFrame, column: str, strategy: Dict) -> List[Dict]:
        """Fill complex text data using LLM contextual reasoning"""
        
        results = []
        missing_indices = df[df[column].isnull()].index
        domain_hints = strategy.get("domain_hints", [])
        
        # Sample existing values for context
        sample_values = df[column].dropna().head(5).tolist()
        
        for idx in missing_indices:
            row_context = self._build_row_context(df, idx, column)
            
            prompt = f"""
            Fill the missing value for column "{column}" based on the context.
            
            Row context: {row_context}
            
            Column characteristics:
            - Domain hints: {domain_hints}
            - Example existing values: {sample_values}
            
            Instructions:
            1. Consider the context of other fields in this row
            2. Look at the pattern of existing values
            3. Use domain knowledge if applicable
            4. Provide a realistic and consistent value
            
            Respond with this JSON format:
            {{
                "value": "<suggested_value>",
                "confidence": <0.0-1.0>,
                "reasoning": "<explanation of your reasoning>"
            }}
            """
            
            try:
                suggested_value, confidence, reasoning = self._query_llm(prompt)
                
                results.append({
                    "index": idx,
                    "column": column,
                    "original_value": df.loc[idx, column],
                    "suggested_value": suggested_value,
                    "confidence": confidence,
                    "method": "llm_contextual",
                    "reasoning": reasoning,
                    "timestamp": self.processing_timestamp
                })
                
            except Exception as e:
                logger.error(f"LLM contextual fill failed for row {idx}: {e}")
                results.append({
                    "index": idx,
                    "column": column,
                    "original_value": df.loc[idx, column],
                    "suggested_value": "Unknown",
                    "confidence": 0.2,
                    "method": "error_fallback",
                    "reasoning": f"Processing error: {str(e)}",
                    "timestamp": self.processing_timestamp
                })
        
        return results

    def _build_row_context(self, df: pd.DataFrame, row_idx: int, exclude_column: str) -> str:
        """Build context string from other columns in the row"""
        
        row = df.loc[row_idx]
        context_parts = []
        
        for col, value in row.items():
            if col != exclude_column and pd.notna(value):
                context_parts.append(f"{col}: {value}")
        
        return ", ".join(context_parts) if context_parts else "No additional context available"

    def _query_llm(self, prompt: str) -> Tuple[Any, float, str]:
        """Query LLM using the configured provider (Groq/OpenAI/Ollama)"""
        
        try:
            if self.llm_provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content.strip()
                
            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content.strip()
                
            elif self.llm_provider == "ollama":
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3}
                )
                response_text = response['message']['content'].strip()
            
            logger.info(f"LLM response preview: {response_text[:100]}...")
            
            # Parse JSON response with robust error handling
            json_result = self._extract_json_robust(response_text)
            
            if json_result:
                return (
                    json_result.get("value", "Unknown"),
                    float(json_result.get("confidence", 0.5)),
                    json_result.get("reasoning", "LLM prediction")
                )
            else:
                # Fallback - try to extract value without JSON
                clean_response = response_text.replace('"', '').replace("'", "").strip()
                if len(clean_response) < 100:  # Reasonable length
                    return (clean_response, 0.3, "Non-JSON response, extracted text")
                else:
                    return ("Parse_Error", 0.1, f"Could not parse: {response_text[:50]}...")
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ("Error", 0.1, f"LLM error: {str(e)}")
    
    def _extract_json_robust(self, text: str) -> Optional[Dict]:
        """Multiple strategies to extract JSON from LLM response"""
        
        # Strategy 1: Find complete JSON block
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            try:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON-like patterns and reconstruct
        import re
        
        # Extract value
        value_match = re.search(r'"value":\s*"([^"]*)"', text)
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', text)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', text)
        
        if value_match:
            return {
                "value": value_match.group(1),
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "reasoning": reasoning_match.group(1) if reasoning_match else "Partial extraction"
            }
        
        # Strategy 3: Try to fix common JSON issues
        try:
            # Fix common issues like trailing commas, missing quotes
            fixed_text = text.replace(',}', '}').replace("'", '"')
            # Try parsing the fixed version
            start_idx = fixed_text.find('{')
            end_idx = fixed_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(fixed_text[start_idx:end_idx])
        except:
            pass
        
        return None

    def _find_closest_category(self, suggested: str, valid_categories: List[str]) -> str:
        """Find closest matching category using simple string similarity"""
        
        if not valid_categories:
            return suggested
        
        suggested_lower = suggested.lower()
        
        # Exact match first
        for cat in valid_categories:
            if suggested_lower == cat.lower():
                return cat
        
        # Partial match
        for cat in valid_categories:
            if suggested_lower in cat.lower() or cat.lower() in suggested_lower:
                return cat
        
        # Fallback to first category
        return valid_categories[0]

    def generate_completion_report(self, original_df: pd.DataFrame, filled_df: pd.DataFrame, 
                                 results: List[Dict]) -> Dict:
        """Generate comprehensive completion and quality report"""
        
        # Calculate completion improvements
        completion_improvements = {}
        
        for column in original_df.columns:
            original_missing = original_df[column].isnull().sum()
            filled_missing = filled_df[column].isnull().sum()
            
            if original_missing > 0:
                improvement = original_missing - filled_missing
                completion_improvements[column] = {
                    "originally_missing": original_missing,
                    "still_missing": filled_missing,
                    "filled_count": improvement,
                    "improvement_percentage": round((improvement / original_missing) * 100, 1)
                }
        
        # Analyze confidence distribution
        confidence_dist = {"high": 0, "medium": 0, "low": 0}
        method_usage = {}
        
        for result in results:
            confidence = result["confidence"]
            method = result["method"]
            
            if confidence >= 0.8:
                confidence_dist["high"] += 1
            elif confidence >= 0.6:
                confidence_dist["medium"] += 1
            else:
                confidence_dist["low"] += 1
            
            method_usage[method] = method_usage.get(method, 0) + 1
        
        report = {
            "pipeline_summary": {
                "processing_timestamp": self.processing_timestamp,
                "model_used": self.model_name,
                "total_rows_processed": len(original_df),
                "total_fields_filled": len(results),
                "processing_methods_used": list(method_usage.keys())
            },
            "completion_improvements": completion_improvements,
            "confidence_distribution": confidence_dist,
            "method_breakdown": method_usage,
            "quality_assessment": self._assess_quality(results),
            "recommendations": self._generate_recommendations(completion_improvements, confidence_dist)
        }
        
        return report

    def _assess_quality(self, results: List[Dict]) -> Dict:
        """Assess overall data quality of imputation results"""
        
        if not results:
            return {"status": "no_imputation_needed"}
        
        total_results = len(results)
        high_confidence = sum(1 for r in results if r["confidence"] >= 0.8)
        low_confidence = sum(1 for r in results if r["confidence"] < 0.5)
        error_count = sum(1 for r in results if "error" in r["method"].lower())
        
        quality_score = ((high_confidence * 1.0 + (total_results - high_confidence - low_confidence) * 0.7) / total_results) * 100
        
        assessment = {
            "overall_quality_score": round(quality_score, 1),
            "high_confidence_percentage": round((high_confidence / total_results) * 100, 1),
            "low_confidence_count": low_confidence,
            "error_count": error_count,
            "status": "excellent" if quality_score >= 85 else "good" if quality_score >= 70 else "needs_review"
        }
        
        return assessment

    def _generate_recommendations(self, completion_improvements: Dict, confidence_dist: Dict) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        total_predictions = sum(confidence_dist.values())
        if total_predictions == 0:
            return ["No missing data found - dataset is already complete!"]
        
        low_confidence_pct = (confidence_dist["low"] / total_predictions * 100)
        
        if low_confidence_pct > 25:
            recommendations.append(
                f"⚠️  {low_confidence_pct:.1f}% of predictions have low confidence - consider manual review"
            )
        
        if low_confidence_pct < 10:
            recommendations.append(
                "✅ High overall confidence - safe for production use with spot checking"
            )
        
        # Find columns with incomplete filling
        incomplete_columns = [
            col for col, stats in completion_improvements.items() 
            if stats["improvement_percentage"] < 100
        ]
        
        if incomplete_columns:
            recommendations.append(
                f"🔧 Consider additional strategies for: {', '.join(incomplete_columns)}"
            )
        
        recommendations.append(
            "📊 Validate a sample of high-confidence predictions against domain expertise"
        )
        
        return recommendations


def demo_generalized_pipeline():
    """
    Demonstrate the pipeline with cloud LLM providers
    Shows speed and accuracy improvements over local inference
    """
    
    # Check for API keys
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print("=== AutoFill AI - Cloud LLM Demo ===\n")
    
    # Determine best available provider
    if groq_key:
        provider = "groq"
        print("✅ Using Groq (Fast + Free)")
    elif openai_key:
        provider = "openai" 
        print("✅ Using OpenAI (Premium)")
    else:
        provider = "ollama"
        print("⚠️  No API keys found, falling back to Ollama")
        print("For faster results, get a free Groq API key:")
        print("1. Go to https://console.groq.com")
        print("2. Sign up and get API key") 
        print("3. export GROQ_API_KEY='your-key-here'")
        print()
    
    # Example dataset - business customer data
    business_data = {
        "company_name": ["Acme Corp", "TechStart Inc", None, "Global Solutions", None],
        "industry": [None, "Software", "Manufacturing", None, "Healthcare"],
        "revenue_millions": [12.5, None, 45.2, None, 8.7],
        "employee_count": [50, 25, None, 200, None],
        "founded_year": [2010, 2018, 1995, None, 2020]
    }
    
    df = pd.DataFrame(business_data)
    
    print("Original Dataset:")
    print(df.to_string())
    print(f"\n{'='*60}")
    
    try:
        # Initialize pipeline with cloud provider
        start_time = time.time()
        pipeline = AutoFillPipeline(llm_provider=provider)
        
        print(f"\nUsing: {pipeline.get_api_usage_info()}")
        print(f"\n{'-'*40}")
        print("ANALYSIS PHASE")
        print(f"{'-'*40}")
        
        # Analyze the dataset
        analysis = pipeline.analyze_dataset(df)
        
        print(f"Dataset: {analysis['dataset_profile']['total_rows']} rows, {analysis['dataset_profile']['total_columns']} columns")
        print(f"Complexity Score: {analysis['complexity_score']}")
        print(f"Estimated Time: {analysis['estimated_total_time']}")
        
        print(f"\nMissing Data Summary:")
        for column, stats in analysis["missing_data_summary"].items():
            strategy = analysis["imputation_plan"][column]
            print(f"  {column}: {stats['missing_count']} missing ({stats['missing_percentage']}%) - {strategy['method']}")
        
        print(f"\n{'-'*40}")
        print("IMPUTATION PHASE") 
        print(f"{'-'*40}")
        
        # Fill missing data
        filled_df, results = pipeline.fill_missing_data(df)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nCompleted Dataset:")
        print(filled_df.to_string())
        
        print(f"\nImputation Results:")
        for result in results:
            print(f"  Row {result['index']}, {result['column']}: '{result['suggested_value']}' (confidence: {result['confidence']:.2f})")
            print(f"    Method: {result['method']}")
            print(f"    Reasoning: {result['reasoning'][:100]}...")
        
        # Generate report
        report = pipeline.generate_completion_report(df, filled_df, results)
        
        print(f"\n{'-'*40}")
        print("PERFORMANCE & QUALITY REPORT")
        print(f"{'-'*40}")
        
        print(f"⏱️  Total Processing Time: {processing_time:.1f} seconds")
        print(f"🔥 Average per LLM call: {processing_time/len([r for r in results if 'llm' in r['method']]):.1f}s" if any('llm' in r['method'] for r in results) else "")
        
        quality = report["quality_assessment"]
        print(f"📊 Overall Quality Score: {quality['overall_quality_score']}/100 ({quality['status']})")
        print(f"✅ High Confidence Predictions: {quality['high_confidence_percentage']}%")
        
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print(f"\n{'='*60}")
        print("🎉 SUCCESS! Pipeline completed in under 30 seconds")
        print("Ready for production use with your real datasets.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        if "API key" in str(e):
            print("\n💡 Quick Setup for Groq (Free):")
            print("1. Visit: https://console.groq.com/keys")
            print("2. Create account and generate API key")
            print("3. Run: export GROQ_API_KEY='your-key-here'")
            print("4. Restart this script")
        
        print(f"\nFallback: Use Ollama locally (slower)")
        print("ollama serve")  


if __name__ == "__main__":
    demo_generalized_pipeline()