# src/stage2/validation.py
"""
Stage 2 validation utilities for prompt generation and response validation.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError


logger = get_logger(__name__)


class Stage2Validator:
    """Comprehensive validation for Stage 2 prompt generation and responses."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_stage1_inputs(self, stage1_data: Dict[str, Any]) -> bool:
        """Validate Stage 1 data for Stage 2 processing."""
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Check required top-level keys
        required_keys = ['metadata', 'customer_archetypes', 'execution_queries']
        for key in required_keys:
            if key not in stage1_data:
                self.validation_errors.append(f"Missing required key: {key}")
        
        # Validate metadata
        if 'metadata' in stage1_data:
            metadata = stage1_data['metadata']
            required_metadata = ['project_id', 'brand', 'category', 'correlation_id']
            for key in required_metadata:
                if key not in metadata:
                    self.validation_errors.append(f"Missing metadata key: {key}")
        
        # Validate customer archetypes
        if 'customer_archetypes' in stage1_data:
            archetypes = stage1_data['customer_archetypes']
            if not isinstance(archetypes, list) or len(archetypes) == 0:
                self.validation_errors.append("No customer archetypes found")
            else:
                for i, archetype in enumerate(archetypes):
                    self._validate_archetype_structure(archetype, i)
        
        # Validate execution queries
        if 'execution_queries' in stage1_data:
            queries = stage1_data['execution_queries']
            if not isinstance(queries, list) or len(queries) == 0:
                self.validation_errors.append("No execution queries found")
            else:
                for i, query in enumerate(queries):
                    self._validate_query_structure(query, i)
        
        # Log validation results
        if self.validation_errors:
            logger.error("Stage 1 input validation failed", 
                        metadata={"errors": self.validation_errors})
            return False
        
        if self.validation_warnings:
            logger.warning("Stage 1 input validation warnings", 
                          metadata={"warnings": self.validation_warnings})
        
        logger.info("Stage 1 input validation passed")
        return True
    
    def _validate_archetype_structure(self, archetype: Dict[str, Any], index: int) -> None:
        """Validate individual archetype structure."""
        required_fields = ['archetype_id', 'name', 'description']
        for field in required_fields:
            if field not in archetype:
                self.validation_errors.append(f"Archetype {index}: missing {field}")
        
        # Check for attributes
        if 'attributes' not in archetype:
            self.validation_warnings.append(f"Archetype {index}: missing attributes")
        
        # Check for strategic fields
        strategic_fields = ['market_presence', 'strategic_value', 'confidence']
        for field in strategic_fields:
            if field not in archetype:
                self.validation_warnings.append(f"Archetype {index}: missing {field}")
    
    def _validate_query_structure(self, query: Dict[str, Any], index: int) -> None:
        """Validate individual query structure."""
        required_fields = ['query_id', 'styled_query']
        for field in required_fields:
            if field not in query:
                self.validation_errors.append(f"Query {index}: missing {field}")
        
        # Check query quality indicators
        quality_fields = ['archetype', 'category', 'authenticity_score']
        for field in quality_fields:
            if field not in query:
                self.validation_warnings.append(f"Query {index}: missing {field}")
        
        # Validate authenticity score
        if 'authenticity_score' in query:
            score = query['authenticity_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                self.validation_warnings.append(f"Query {index}: invalid authenticity_score")
    
    def validate_generated_prompts(self, natural_prompts: List[Dict], controlled_prompts: List[Dict]) -> bool:
        """Validate generated prompt structures."""
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Check prompt counts match
        if len(natural_prompts) != len(controlled_prompts):
            self.validation_errors.append(
                f"Prompt count mismatch: {len(natural_prompts)} natural vs {len(controlled_prompts)} controlled"
            )
        
        # Validate natural prompts
        for i, prompt in enumerate(natural_prompts):
            self._validate_prompt_structure(prompt, i, "natural")
        
        # Validate controlled prompts
        for i, prompt in enumerate(controlled_prompts):
            self._validate_prompt_structure(prompt, i, "controlled")
        
        # Check for prompt ID uniqueness
        all_prompt_ids = [p.get('prompt_id') for p in natural_prompts + controlled_prompts]
        unique_ids = set(all_prompt_ids)
        if len(all_prompt_ids) != len(unique_ids):
            self.validation_errors.append("Duplicate prompt IDs found")
        
        if self.validation_errors:
            logger.error("Prompt validation failed", 
                        metadata={"errors": self.validation_errors})
            return False
        
        if self.validation_warnings:
            logger.warning("Prompt validation warnings", 
                          metadata={"warnings": self.validation_warnings})
        
        logger.info("Prompt validation passed")
        return True
    
    def _validate_prompt_structure(self, prompt: Dict[str, Any], index: int, prompt_type: str) -> None:
        """Validate individual prompt structure."""
        required_fields = [
            'prompt_id', 'query_id', 'query_text', 'prompt_type', 
            'formatted_prompt', 'platforms', 'analysis_focus'
        ]
        
        for field in required_fields:
            if field not in prompt:
                self.validation_errors.append(f"{prompt_type} prompt {index}: missing {field}")
        
        # Validate prompt type
        if prompt.get('prompt_type') != f"{prompt_type}_ai":
            self.validation_errors.append(f"{prompt_type} prompt {index}: incorrect prompt_type")
        
        # Validate platforms
        if 'platforms' in prompt:
            platforms = prompt['platforms']
            expected_platforms = ["claude", "chatgpt", "gemini", "grok"]
            if not isinstance(platforms, list) or set(platforms) != set(expected_platforms):
                self.validation_warnings.append(f"{prompt_type} prompt {index}: unexpected platforms list")
        
        # Validate formatted prompt content
        if 'formatted_prompt' in prompt:
            formatted = prompt['formatted_prompt']
            if not isinstance(formatted, str) or len(formatted) < 100:
                self.validation_warnings.append(f"{prompt_type} prompt {index}: formatted_prompt seems too short")
            
            # Check for template variables that weren't replaced
            if '{' in formatted and '}' in formatted:
                self.validation_warnings.append(f"{prompt_type} prompt {index}: unreplaced template variables detected")
    
    def validate_execution_package(self, execution_package: Dict[str, Any]) -> bool:
        """Validate complete execution package structure."""
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Check required top-level sections
        required_sections = [
            'metadata', 'execution_summary', 'query_prompt_mapping', 
            'natural_prompts', 'controlled_prompts', 'file_structure'
        ]
        
        for section in required_sections:
            if section not in execution_package:
                self.validation_errors.append(f"Missing execution package section: {section}")
        
        # Validate execution summary
        if 'execution_summary' in execution_package:
            summary = execution_package['execution_summary']
            required_summary_fields = [
                'total_queries', 'total_prompts', 'platforms', 'expected_responses'
            ]
            
            for field in required_summary_fields:
                if field not in summary:
                    self.validation_errors.append(f"Missing execution summary field: {field}")
            
            # Validate mathematical consistency
            total_queries = summary.get('total_queries', 0)
            total_prompts = summary.get('total_prompts', 0)
            expected_responses = summary.get('expected_responses', 0)
            platforms = summary.get('platforms', [])
            
            if total_prompts != total_queries * 2:  # natural + controlled
                self.validation_warnings.append("Total prompts doesn't equal queries * 2")
            
            if expected_responses != total_queries * len(platforms) * 2:
                self.validation_warnings.append("Expected responses calculation seems incorrect")
        
        # Validate query-prompt mappings
        if 'query_prompt_mapping' in execution_package:
            mappings = execution_package['query_prompt_mapping']
            if not isinstance(mappings, list):
                self.validation_errors.append("query_prompt_mapping should be a list")
            else:
                for i, mapping in enumerate(mappings):
                    self._validate_query_mapping(mapping, i)
        
        if self.validation_errors:
            logger.error("Execution package validation failed", 
                        metadata={"errors": self.validation_errors})
            return False
        
        if self.validation_warnings:
            logger.warning("Execution package validation warnings", 
                          metadata={"warnings": self.validation_warnings})
        
        logger.info("Execution package validation passed")
        return True
    
    def _validate_query_mapping(self, mapping: Dict[str, Any], index: int) -> None:
        """Validate individual query-prompt mapping."""
        required_fields = [
            'query_id', 'query_text', 'natural_prompt_id', 
            'controlled_prompt_id', 'execution_priority', 'expected_files'
        ]
        
        for field in required_fields:
            if field not in mapping:
                self.validation_errors.append(f"Query mapping {index}: missing {field}")
        
        # Validate expected files
        if 'expected_files' in mapping:
            expected_files = mapping['expected_files']
            if not isinstance(expected_files, list) or len(expected_files) == 0:
                self.validation_errors.append(f"Query mapping {index}: invalid expected_files")
            else:
                # Check file naming convention
                for file_name in expected_files:
                    if not self._validate_response_filename(file_name):
                        self.validation_warnings.append(
                            f"Query mapping {index}: invalid filename format: {file_name}"
                        )
    
    def _validate_response_filename(self, filename: str) -> bool:
        """Validate response filename follows convention."""
        # Expected pattern: platform_queryNN_type.json
        pattern = r'^(claude|chatgpt|gemini|grok)_query\d{2}_(natural|controlled)\.json$'
        return re.match(pattern, filename) is not None
    
    def validate_manual_responses(self, responses_dir: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate uploaded manual response files."""
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        if not responses_dir.exists():
            self.validation_errors.append(f"Responses directory not found: {responses_dir}")
            return False, {}
        
        # Find all JSON files
        natural_dir = responses_dir / "natural_dataset"
        controlled_dir = responses_dir / "controlled_dataset"
        
        natural_files = list(natural_dir.glob("*.json")) if natural_dir.exists() else []
        controlled_files = list(controlled_dir.glob("*.json")) if controlled_dir.exists() else []
        
        all_files = natural_files + controlled_files
        
        if not all_files:
            self.validation_errors.append("No response files found")
            return False, {}
        
        validation_stats = {
            "total_files": len(all_files),
            "valid_files": 0,
            "invalid_files": 0,
            "natural_responses": len(natural_files),
            "controlled_responses": len(controlled_files),
            "platforms_found": set(),
            "queries_found": set(),
            "file_errors": []
        }
        
        # Validate each file
        for file_path in all_files:
            try:
                # Check filename format
                if not self._validate_response_filename(file_path.name):
                    self.validation_warnings.append(f"Invalid filename format: {file_path.name}")
                
                # Parse filename for metadata
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 3:
                    platform = filename_parts[0]
                    query_part = filename_parts[1]
                    response_type = filename_parts[2]
                    
                    validation_stats["platforms_found"].add(platform)
                    validation_stats["queries_found"].add(query_part)
                
                # Validate JSON structure
                with open(file_path, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                
                # Basic structure validation
                if self._validate_response_structure(response_data, file_path.name):
                    validation_stats["valid_files"] += 1
                else:
                    validation_stats["invalid_files"] += 1
                    validation_stats["file_errors"].append(file_path.name)
                
            except json.JSONDecodeError as e:
                self.validation_errors.append(f"Invalid JSON in {file_path.name}: {e}")
                validation_stats["invalid_files"] += 1
                validation_stats["file_errors"].append(file_path.name)
            
            except Exception as e:
                self.validation_errors.append(f"Error processing {file_path.name}: {e}")
                validation_stats["invalid_files"] += 1
                validation_stats["file_errors"].append(file_path.name)
        
        # Check coverage
        expected_platforms = {"claude", "chatgpt", "gemini", "grok"}
        missing_platforms = expected_platforms - validation_stats["platforms_found"]
        if missing_platforms:
            self.validation_warnings.append(f"Missing platforms: {missing_platforms}")
        
        success = validation_stats["valid_files"] > 0 and len(self.validation_errors) == 0
        
        if not success:
            logger.error("Manual response validation failed", 
                        metadata={"errors": self.validation_errors, "stats": validation_stats})
        else:
            logger.info("Manual response validation passed", 
                       metadata={"stats": validation_stats})
        
        return success, validation_stats
    
    def _validate_response_structure(self, response_data: Dict[str, Any], filename: str) -> bool:
        """Validate individual response file structure."""
        
        # Check for required top-level keys
        required_keys = ["response_metadata", "response_content"]
        for key in required_keys:
            if key not in response_data:
                self.validation_warnings.append(f"{filename}: missing {key}")
                return False
        
        # Validate metadata structure
        metadata = response_data.get("response_metadata", {})
        required_metadata = ["platform", "query_id", "prompt_type", "timestamp"]
        for key in required_metadata:
            if key not in metadata:
                self.validation_warnings.append(f"{filename}: missing metadata.{key}")
        
        # Validate response content
        content = response_data.get("response_content", "")
        if not isinstance(content, str) or len(content) < 10:
            self.validation_warnings.append(f"{filename}: response_content too short or missing")
        
        return True
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "error_count": len(self.validation_errors),
            "warning_count": len(self.validation_warnings),
            "validation_passed": len(self.validation_errors) == 0
        }


# Convenience functions
def validate_stage1_for_stage2(stage1_data: Dict[str, Any]) -> bool:
    """Quick validation of Stage 1 data for Stage 2 processing."""
    validator = Stage2Validator()
    return validator.validate_stage1_inputs(stage1_data)


def validate_manual_responses(responses_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Quick validation of manual response files."""
    validator = Stage2Validator()
    return validator.validate_manual_responses(responses_dir)