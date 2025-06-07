# src/stage2/prompt_executor.py
"""
Stage 2: Natural and Controlled AI Prompt Generation and Execution
UPDATED with prompt template variable injection system.
"""
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from ..core.project_manager import BrandAuditProject
from ..models.project import StageStatusEnum
from ..utils.logging import get_logger
from ..utils.exceptions import StageExecutionError

# NEW IMPORTS for the injection system
from .stage1_data_loader import Stage1DataLoader
from .prompt_template_injector import generate_injectable_prompts


logger = get_logger(__name__)


class Stage2PromptExecutor:
    """
    Stage 2 orchestrator that generates natural and controlled AI prompts
    with full execution packages and maintains JSON traceability.
    
    UPDATED: Now uses prompt template variable injection system.
    """
    
    def __init__(self, project: BrandAuditProject):
        self.project = project
        self.correlation_id = str(uuid4())
        
        # NEW: Initialize the Stage 1 data loader
        self.stage1_loader = Stage1DataLoader(project)
        
        logger.set_context(
            correlation_id=self.correlation_id,
            project_id=project.project_id,
            operation="stage2_execution"
        )
    
    async def execute_prompt_generation(self) -> Dict[str, Any]:
        """
        Execute Stage 2 prompt generation pipeline.
        UPDATED: Now uses the prompt injection system instead of manual prompt assembly.
        
        Returns:
            Dict with generated prompts, execution packages, and traceability data
        """
        try:
            logger.info("Starting Stage 2 prompt generation")
            
            # Update project status
            self.project.update_stage_status("stage2", StageStatusEnum.IN_PROGRESS)
            
            # Step 1: Load Stage 1 outputs using the new data loader
            logger.info("Loading Stage 1 outputs")
            stage1_data = self.stage1_loader.load_stage1_outputs()
            
            # Step 2: Select priority queries for execution
            logger.info("Selecting priority queries")
            priority_queries = self.stage1_loader.get_priority_queries(stage1_data, max_queries=5)
            
            # Step 3: Extract customer context
            logger.info("Extracting customer context")
            customer_context = self.stage1_loader.extract_customer_context(stage1_data)
            
            # Step 4: Generate injectable prompts using the new system
            logger.info("Generating injectable prompts and execution guide")
            execution_package, execution_guide = generate_injectable_prompts(
                stage1_data=stage1_data,
                priority_queries=priority_queries,
                customer_context=customer_context,
                project_metadata=stage1_data.metadata
            )
            
            # Step 5: Save all artifacts with traceability
            logger.info("Saving Stage 2 artifacts")
            artifacts = self._save_stage2_artifacts(execution_package, execution_guide)
            
            # Step 6: Update project status
            self.project.update_stage_status(
                "stage2",
                StageStatusEnum.COMPLETE,
                outputs=list(artifacts.keys())
            )
            
            logger.info("Stage 2 prompt generation completed successfully",
                       metadata={
                           "artifacts_count": len(artifacts),
                           "prompts_generated": execution_package['metadata']['prompts_generated'],
                           "expected_files": execution_package['execution_summary']['expected_response_files']
                       })
            
            return {
                'status': 'success',
                'artifacts': artifacts,
                'execution_package': execution_package,
                'execution_guide': execution_guide,
                'priority_queries_count': len(priority_queries),
                'next_steps': 'manual_execution'
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error("Stage 2 prompt generation failed", 
                        metadata={"error": error_message})
            
            self._update_failed_status(error_message)
            
            raise StageExecutionError(
                f"Stage 2 execution failed: {error_message}",
                stage="stage2",
                operation="prompt_generation"
            ) from e
        finally:
            logger.clear_context()
    
    def _save_stage2_artifacts(
        self, 
        execution_package: Dict[str, Any], 
        execution_guide: str
    ) -> Dict[str, str]:
        """
        Save all Stage 2 artifacts with proper organization.
        UPDATED: Now saves the new execution package and guide format.
        
        Args:
            execution_package: Complete execution package with injectable prompts
            execution_guide: Human-readable execution guide in markdown
            
        Returns:
            Dict mapping artifact names to file paths
        """
        stage2_dir = self.project.get_file_path("stage2_execution")
        stage2_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for manual execution
        (stage2_dir / "natural_dataset").mkdir(exist_ok=True)
        (stage2_dir / "controlled_dataset").mkdir(exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%m%d%y_%H%M")
        
        # Save execution package (machine-readable)
        package_file = stage2_dir / f"execution_package_{timestamp}.json"
        with open(package_file, 'w', encoding='utf-8') as f:
            json.dump(execution_package, f, indent=2, default=str)
        saved_files['execution_package'] = str(package_file)
        
        # Save execution guide (human-readable)
        guide_file = stage2_dir / f"manual_execution_guide_{timestamp}.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(execution_guide)
        saved_files['execution_guide'] = str(guide_file)
        
        # Save individual prompt files for easy access
        prompts_dir = stage2_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        # Save natural prompts
        natural_file = prompts_dir / f"natural_prompts_{timestamp}.json"
        with open(natural_file, 'w', encoding='utf-8') as f:
            json.dump(execution_package['natural_prompts'], f, indent=2, default=str)
        saved_files['natural_prompts'] = str(natural_file)
        
        # Save controlled prompts
        controlled_file = prompts_dir / f"controlled_prompts_{timestamp}.json"
        with open(controlled_file, 'w', encoding='utf-8') as f:
            json.dump(execution_package['controlled_prompts'], f, indent=2, default=str)
        saved_files['controlled_prompts'] = str(controlled_file)
        
        # Log saved files
        for artifact_name, file_path in saved_files.items():
            file_size = Path(file_path).stat().st_size / 1024  # KB
            logger.info(f"Saved {artifact_name}",
                       metadata={"file": file_path, "size_kb": f"{file_size:.1f}"})
        
        return saved_files
    
    def _update_failed_status(self, error_message: str) -> None:
        """Update project status to failed with error message."""
        try:
            stage_status = self.project.config.stage_status['stage2']
            stage_status.status = StageStatusEnum.FAILED
            stage_status.error_message = error_message
            stage_status.retry_count += 1
            
            self.project.config.project_metadata.last_modified = datetime.now()
            self.project._save_config()
            
            logger.info("Project status updated to FAILED", 
                       metadata={"error_message": error_message})
                       
        except Exception as status_error:
            logger.error("Failed to update project status to FAILED", 
                        metadata={"status_error": str(status_error)})


# Factory function for Stage 2 execution (UPDATED)
async def execute_stage2(project: BrandAuditProject) -> Dict[str, Any]:
    """Execute Stage 2 prompt generation for the given project."""
    executor = Stage2PromptExecutor(project)
    return await executor.execute_prompt_generation()


# LEGACY METHODS REMOVED:
# The following methods are now replaced by the prompt injection system:
# - _load_stage1_outputs() -> replaced by Stage1DataLoader
# - _select_priority_queries() -> replaced by Stage1DataLoader.get_priority_queries()
# - _generate_natural_ai_prompts() -> replaced by prompt injection system
# - _generate_controlled_ai_prompts() -> replaced by prompt injection system
# - _create_execution_packages() -> replaced by ExecutionPackageGenerator
# - _generate_execution_guide() -> replaced by ExecutionGuideGenerator

# These were the manual prompt assembly methods that are now automated.