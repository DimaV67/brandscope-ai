# src/stage2/stage1_data_loader.py
"""
Stage 1 → Stage 2 Data Pipeline
Loads, validates, and transforms Stage 1 outputs for Stage 2 prompt generation.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..utils.exceptions import StageExecutionError, ValidationError
from .validation import Stage2Validator


logger = get_logger(__name__)


@dataclass
class Stage1Data:
    """Structured representation of Stage 1 outputs"""
    metadata: Dict[str, Any]
    customer_archetypes: List[Dict[str, Any]]
    execution_queries: List[Dict[str, Any]]
    framework_compliance: Dict[str, Any]
    cohort_parameters: Dict[str, Any]
    attribute_model: Optional[Dict[str, Any]] = None
    category_context: Optional[Dict[str, Any]] = None


class Stage1DataLoader:
    """
    Loads and validates Stage 1 outputs for Stage 2 processing.
    Handles multiple Stage 1 output formats and provides unified interface.
    """
    
    def __init__(self, project):
        self.project = project
        self.validator = Stage2Validator()
        
    def load_stage1_outputs(self) -> Stage1Data:
        """
        Load complete Stage 1 data with validation and error recovery.
        
        Returns:
            Stage1Data: Validated and structured Stage 1 outputs
            
        Raises:
            StageExecutionError: If Stage 1 data is missing or invalid
        """
        try:
            logger.info("Loading Stage 1 outputs", metadata={"project_id": self.project.project_id})
            
            # Step 1: Locate Stage 1 output files
            stage1_files = self._discover_stage1_files()
            
            # Step 2: Load execution package (primary data source)
            execution_package = self._load_execution_package(stage1_files)
            
            # Step 3: Load supplementary data (attribute model, category context)
            supplementary_data = self._load_supplementary_data(stage1_files)
            
            # Step 4: Validate loaded data
            stage1_data = self._construct_stage1_data(execution_package, supplementary_data)
            
            # Step 5: Validation
            if not self.validator.validate_stage1_inputs(stage1_data.__dict__):
                raise ValidationError(
                    f"Stage 1 data validation failed: {self.validator.validation_errors}"
                )
            
            logger.info("Stage 1 data loaded successfully", 
                       metadata={
                           "archetypes_count": len(stage1_data.customer_archetypes),
                           "queries_count": len(stage1_data.execution_queries),
                           "framework_compliance": stage1_data.framework_compliance.get('framework_coverage', 'unknown')
                       })
            
            return stage1_data
            
        except Exception as e:
            logger.error("Failed to load Stage 1 outputs", 
                        metadata={"error": str(e), "project_id": self.project.project_id})
            raise StageExecutionError(
                f"Cannot load Stage 1 outputs: {str(e)}",
                stage="stage2",
                operation="load_stage1_data"
            ) from e
    
    def _discover_stage1_files(self) -> Dict[str, Path]:
        """
        Discover all relevant Stage 1 output files based on your naming convention.
        Updated to match your Stage1Generator output pattern.
        
        Returns:
            Dict mapping file types to file paths
        """
        stage1_dir = self.project.get_file_path("stage1_outputs")
        
        if not stage1_dir.exists():
            raise StageExecutionError(
                "Stage 1 outputs directory not found. Run Stage 1 first.",
                stage="stage2", 
                operation="discover_stage1_files"
            )
        
        # Find files by your specific pattern matching
        files = {
            # Your Stage1Generator creates these files:
            'category_intelligence': list(stage1_dir.glob("01_category_intelligence_*.json")),
            'archetypes': list(stage1_dir.glob("02_customer_archetypes_*.json")),
            'styled_queries': list(stage1_dir.glob("03_styled_queries_*.json")),
            'execution_packages': list(stage1_dir.glob("04_execution_package_*.json")),
            
            # Legacy patterns for compatibility:
            'attribute_models': list(stage1_dir.glob("*AttributeModel_FINAL*.json")) + list(stage1_dir.glob("*04_*.json")),
            'context_files': list(stage1_dir.glob("*context_for_step*.json")) + list(stage1_dir.glob("*context*.json")),
            'cohort_outputs': list(stage1_dir.glob("*stage1_output*.json")) + list(stage1_dir.glob("*cohort*.json"))
        }
        
        # Select most recent files for each category
        selected_files = {}
        for file_type, file_list in files.items():
            if file_list:
                # Sort by modification time, most recent first
                selected_files[file_type] = max(file_list, key=lambda f: f.stat().st_mtime)
        
        logger.info("Discovered Stage 1 files", 
                   metadata={"files_found": list(selected_files.keys())})
        
        return selected_files
    
    def _load_execution_package(self, stage1_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Load the primary execution package from Stage 1.
        Updated to handle your Stage1Generator output format.
        
        Args:
            stage1_files: Dictionary of discovered Stage 1 files
            
        Returns:
            Dict containing execution package data
        """
        # Priority order for execution package sources based on your Stage 1 output
        package_sources = [
            'execution_packages',  # 04_execution_package_*.json (preferred)
            'styled_queries',      # 03_styled_queries_*.json  
            'cohort_outputs'       # Legacy format fallback
        ]
        
        execution_data = None
        
        for source_type in package_sources:
            if source_type in stage1_files:
                file_path = stage1_files[source_type]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    logger.info(f"Loaded execution package from {source_type}",
                               metadata={"file": str(file_path)})
                    
                    execution_data = self._normalize_execution_package(data, source_type)
                    break
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load {source_type}",
                                 metadata={"file": str(file_path), "error": str(e)})
                    continue
        
        if execution_data is None:
            raise StageExecutionError(
                "No valid execution package found in Stage 1 outputs",
                stage="stage2",
                operation="load_execution_package"
            )
        
        # If we loaded from styled_queries, try to supplement with archetype data
        if execution_data.get('metadata', {}).get('source') != 'execution_packages':
            if 'archetypes' in stage1_files:
                execution_data = self._supplement_with_archetypes(execution_data, stage1_files['archetypes'])
        
        return execution_data
    
    def _normalize_execution_package(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        Normalize different Stage 1 output formats to standard structure.
        
        Args:
            data: Raw data from Stage 1 file
            source_type: Type of source file
            
        Returns:
            Normalized execution package data
        """
        if source_type == 'execution_packages':
            # Already in correct format from your Stage1Generator
            return data
        
        elif source_type == 'styled_queries':
            # Load styled queries from Stage 1 - need to find customer archetypes elsewhere
            return {
                'metadata': data.get('generation_metadata', {}),
                'customer_archetypes': [],  # Will load from archetypes file
                'execution_queries': data.get('styled_queries', []),
                'framework_compliance': data.get('framework_compliance', {}),
                'cohort_parameters': {}
            }
        
        elif source_type == 'cohort_outputs':
            # Fallback for legacy format
            return {
                'metadata': data.get('framework_compliance', {}),
                'customer_archetypes': self._extract_archetypes_from_cohort(data),
                'execution_queries': data.get('filtered_queries', data.get('queries', [])),
                'framework_compliance': data.get('framework_compliance', {}),
                'cohort_parameters': data.get('parameters', {}),
                'cohort_validation': data.get('cohort_validation', {})
            }
        
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _supplement_with_archetypes(self, execution_data: Dict[str, Any], archetypes_file: Path) -> Dict[str, Any]:
        """
        Supplement execution data with archetype information from separate file.
        
        Args:
            execution_data: Basic execution data
            archetypes_file: Path to archetypes JSON file
            
        Returns:
            Enhanced execution data with archetype information
        """
        try:
            with open(archetypes_file, 'r', encoding='utf-8') as f:
                archetypes_data = json.load(f)
            
            # Extract customer archetypes from your format
            execution_data['customer_archetypes'] = archetypes_data.get('ranked_archetypes', [])
            
            # Add archetype metadata
            if 'metadata' not in execution_data:
                execution_data['metadata'] = {}
            
            generation_metadata = archetypes_data.get('generation_metadata', {})
            execution_data['metadata'].update({
                'archetypes_loaded': True,
                'archetype_confidence': generation_metadata.get('avg_confidence', 0.8),
                'archetypes_count': generation_metadata.get('total_archetypes', 0)
            })
            
            logger.info("Supplemented execution data with archetypes",
                       metadata={
                           "archetypes_count": len(execution_data['customer_archetypes']),
                           "file": str(archetypes_file)
                       })
            
        except Exception as e:
            logger.warning("Failed to supplement with archetypes",
                          metadata={"error": str(e), "file": str(archetypes_file)})
        
        return execution_data
    
    def _extract_archetypes_from_cohort(self, cohort_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract customer archetypes from legacy cohort validation data.
        
        Args:
            cohort_data: Raw cohort output data
            
        Returns:
            List of customer archetype dictionaries
        """
        archetypes = []
        
        # Check if explicit archetypes exist
        if 'customer_archetypes' in cohort_data:
            return cohort_data['customer_archetypes']
        
        # Generate archetype from cohort validation (legacy)
        cohort_validation = cohort_data.get('cohort_validation', {})
        if 'extracted_attributes' in cohort_validation:
            archetype = {
                'archetype_id': 'COHORT_001',
                'name': 'Primary Customer Cohort',
                'description': cohort_data.get('scenario_summary', 'Generated from cohort analysis'),
                'attributes': cohort_validation['extracted_attributes'],
                'confidence': self._calculate_archetype_confidence(cohort_validation['extracted_attributes']),
                'strategic_value': 'HIGH',
                'market_presence': 'VALIDATED_COHORT'
            }
            archetypes.append(archetype)
        
        return archetypes
    
    def _calculate_archetype_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence score for generated archetype."""
        if not attributes:
            return 0.0
        
        confidence_scores = [
            attr.get('confidence', 0.0) for attr in attributes.values() 
            if isinstance(attr, dict) and 'confidence' in attr
        ]
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    def _load_supplementary_data(self, stage1_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Load supplementary data (attribute models, context files).
        
        Args:
            stage1_files: Dictionary of discovered Stage 1 files
            
        Returns:
            Dictionary containing supplementary data
        """
        supplementary = {}
        
        # Load attribute model
        if 'attribute_models' in stage1_files:
            try:
                with open(stage1_files['attribute_models'], 'r', encoding='utf-8') as f:
                    supplementary['attribute_model'] = json.load(f)
                logger.info("Loaded attribute model", 
                           metadata={"file": str(stage1_files['attribute_models'])})
            except Exception as e:
                logger.warning("Failed to load attribute model",
                              metadata={"error": str(e)})
        
        # Load category context
        if 'context_files' in stage1_files:
            try:
                with open(stage1_files['context_files'], 'r', encoding='utf-8') as f:
                    supplementary['category_context'] = json.load(f)
                logger.info("Loaded category context",
                           metadata={"file": str(stage1_files['context_files'])})
            except Exception as e:
                logger.warning("Failed to load category context",
                              metadata={"error": str(e)})
        
        return supplementary
    
    def _construct_stage1_data(self, execution_package: Dict[str, Any], 
                             supplementary_data: Dict[str, Any]) -> Stage1Data:
        """
        Construct unified Stage1Data object from loaded components.
        
        Args:
            execution_package: Primary execution package data
            supplementary_data: Additional context and attribute data
            
        Returns:
            Stage1Data object with all loaded information
        """
        return Stage1Data(
            metadata=execution_package.get('metadata', {}),
            customer_archetypes=execution_package.get('customer_archetypes', []),
            execution_queries=execution_package.get('execution_queries', []),
            framework_compliance=execution_package.get('framework_compliance', {}),
            cohort_parameters=execution_package.get('cohort_parameters', {}),
            attribute_model=supplementary_data.get('attribute_model'),
            category_context=supplementary_data.get('category_context')
        )
    
    def get_priority_queries(self, stage1_data: Stage1Data, max_queries: int = 5) -> List[Dict[str, Any]]:
        """
        Extract priority queries for Stage 2 execution.
        
        Args:
            stage1_data: Loaded Stage 1 data
            max_queries: Maximum number of queries to return
            
        Returns:
            List of priority queries for execution
        """
        queries = stage1_data.execution_queries
        
        if not queries:
            raise StageExecutionError(
                "No execution queries found in Stage 1 data",
                stage="stage2",
                operation="get_priority_queries"
            )
        
        # Sort by priority indicators
        sorted_queries = sorted(
            queries,
            key=lambda q: (
                -q.get('authenticity_score', 0),  # Higher authenticity first
                -q.get('relevance_score', 0),     # Higher relevance first
                q.get('execution_priority', 999)  # Lower priority number first
            )
        )
        
        # Select top queries
        priority_queries = sorted_queries[:max_queries]
        
        logger.info("Selected priority queries",
                   metadata={
                       "total_available": len(queries),
                       "selected_count": len(priority_queries),
                       "avg_authenticity": sum(q.get('authenticity_score', 0) for q in priority_queries) / len(priority_queries)
                   })
        
        return priority_queries
    
    def extract_customer_context(self, stage1_data: Stage1Data) -> str:
        """
        Extract customer context string for prompt generation.
        Updated to work with your Stage1Generator output format.
        
        Args:
            stage1_data: Loaded Stage 1 data
            
        Returns:
            Formatted customer context string
        """
        # Check for customer_context from your execution package
        if hasattr(stage1_data, 'execution_queries') and stage1_data.execution_queries:
            execution_package = stage1_data.__dict__
            if 'customer_context' in execution_package:
                return execution_package['customer_context']
        
        # Check for explicit scenario summary in cohort parameters
        if stage1_data.cohort_parameters and 'scenario_summary' in stage1_data.cohort_parameters:
            return stage1_data.cohort_parameters['scenario_summary']
        
        # Generate from archetype data
        if stage1_data.customer_archetypes:
            primary_archetype = stage1_data.customer_archetypes[0]
            return self._generate_context_from_archetype(primary_archetype)
        
        # Generate from cohort parameters
        if stage1_data.cohort_parameters:
            return self._generate_context_from_parameters(stage1_data.cohort_parameters)
        
        # Fallback
        return "Generic customer seeking product recommendations"
    
    def _generate_context_from_archetype(self, archetype: Dict[str, Any]) -> str:
        """Generate customer context from archetype data."""
        context_parts = []
        
        # Basic description
        if 'description' in archetype:
            context_parts.append(archetype['description'])
        
        # Attribute-based context
        attributes = archetype.get('attributes', {})
        if attributes:
            motivation = self._extract_attribute_value(attributes, 'COREB1')
            urgency = self._extract_attribute_value(attributes, 'COREA2')
            budget = self._extract_attribute_value(attributes, 'MODIFIERD3')
            
            context_parts.extend([
                f"Primary motivation: {motivation}" if motivation else None,
                f"Shopping urgency: {urgency}" if urgency else None,
                f"Budget range: {budget}" if budget else None
            ])
        
        return ". ".join(filter(None, context_parts))
    
    def _generate_context_from_parameters(self, parameters: Dict[str, Any]) -> str:
        """Generate customer context from cohort parameters."""
        context_parts = []
        
        # Extract key parameters
        for key, value in parameters.items():
            if key in ['TRIGGER_SITUATION', 'DECISION_TIMEFRAME', 'ADDITIONAL_CONTEXT']:
                if value:
                    context_parts.append(f"{key.lower().replace('_', ' ')}: {value}")
        
        return ". ".join(context_parts) if context_parts else "Customer seeking product recommendations"
    
    def _extract_attribute_value(self, attributes: Dict[str, Any], attribute_code: str) -> Optional[str]:
        """Extract attribute value from attributes dictionary."""
        if attribute_code in attributes:
            attr_data = attributes[attribute_code]
            if isinstance(attr_data, dict):
                return attr_data.get('value')
            else:
                return str(attr_data)
        return None


# Integration with existing Stage 2 executor
def load_and_validate_stage1_data(project) -> Stage1Data:
    """
    Convenience function to load Stage 1 data for Stage 2 processing.
    
    Args:
        project: BrandAuditProject instance
        
    Returns:
        Stage1Data: Validated Stage 1 outputs ready for Stage 2
        
    Raises:
        StageExecutionError: If Stage 1 data cannot be loaded or validated
    """
    loader = Stage1DataLoader(project)
    return loader.load_stage1_outputs()


# Update to Stage2PromptExecutor class
class Stage2PromptExecutorUpdated:
    """Updated Stage 2 executor with proper Stage 1 data loading."""
    
    def __init__(self, project):
        self.project = project
        self.stage1_loader = Stage1DataLoader(project)
        
    async def execute_prompt_generation(self) -> Dict[str, Any]:
        """Execute Stage 2 with proper Stage 1 data loading."""
        try:
            # Step 1: Load and validate Stage 1 outputs
            stage1_data = self.stage1_loader.load_stage1_outputs()
            
            # Step 2: Extract priority queries
            priority_queries = self.stage1_loader.get_priority_queries(stage1_data)
            
            # Step 3: Extract customer context
            customer_context = self.stage1_loader.extract_customer_context(stage1_data)
            
            # Step 4: Generate prompts with validated data
            natural_prompts = self._generate_natural_ai_prompts(
                priority_queries, stage1_data, customer_context
            )
            
            controlled_prompts = self._generate_controlled_ai_prompts(
                priority_queries, stage1_data, customer_context  
            )
            
            # Continue with existing prompt generation logic...
            
        except Exception as e:
            logger.error("Stage 2 execution failed", metadata={"error": str(e)})
            raise