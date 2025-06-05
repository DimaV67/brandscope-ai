"""
Project management with security, validation, and performance optimization.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.project import ProjectConfig, ProjectMetadata, StageStatus, StageStatusEnum
from ..models.brand import BrandContext
from ..utils.config import get_config
from ..utils.exceptions import (
    ProjectError, ProjectNotFoundError, ProjectExistsError, 
    ProjectCorruptedError, FileOperationError, SecurityError
)
from ..utils.logging import get_logger
from ..utils.security import SecurityValidator


logger = get_logger(__name__)


class ProjectManager:
    """Secure project management with validation and error handling."""
    
    def __init__(self):
        self.config = get_config()
        self.security = SecurityValidator(self.config.security)
        self.projects_root = self.config.projects_root
        self.index_file = self.projects_root / ".projects_index.json"
        self._ensure_projects_directory()
    
    def _ensure_projects_directory(self) -> None:
        """Ensure projects directory structure exists."""
        try:
            self.projects_root.mkdir(parents=True, exist_ok=True)
            
            # Create index file if it doesn't exist
            if not self.index_file.exists():
                self._save_index({})
                
            # Create .gitkeep
            gitkeep = self.projects_root / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
                
            logger.info("Projects directory initialized", 
                       metadata={"projects_root": str(self.projects_root)})
                       
        except Exception as e:
            logger.error("Failed to initialize projects directory", 
                        metadata={"error": str(e)})
            raise FileOperationError(
                "Failed to initialize projects directory",
                file_path=str(self.projects_root),
                operation="directory_creation"
            ) from e
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load projects index with error handling."""
        try:
            if not self.index_file.exists():
                return {}
            
            with open(self.index_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return {}
                return json.loads(content)
                
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load projects index", 
                        metadata={"index_file": str(self.index_file), "error": str(e)})
            # Backup corrupted index
            backup_name = f".projects_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                shutil.copy2(self.index_file, self.projects_root / backup_name)
            except OSError:
                pass
            return {}
    
    def _save_index(self, index: Dict[str, Dict]) -> None:
        """Save projects index with atomic operation."""
        temp_file = self.index_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            
            # Atomic replace
            temp_file.replace(self.index_file)
            
        except OSError as e:
            logger.error("Failed to save projects index", 
                        metadata={"error": str(e)})
            if temp_file.exists():
                temp_file.unlink()
            raise FileOperationError(
                "Failed to save projects index",
                file_path=str(self.index_file),
                operation="index_save"
            ) from e
    
    def _generate_project_id(self, brand: str, category: str) -> str:
        """Generate secure project ID."""
        # Sanitize inputs
        brand_clean = self.security.sanitize_input(brand).lower()
        category_clean = self.security.sanitize_input(category).lower()
        
        # Generate base ID
        base_id = f"{brand_clean}_{category_clean}_{datetime.now().strftime('%Y%m%d')}"
        
        # Ensure uniqueness
        counter = 0
        project_id = base_id
        while self._project_exists(project_id):
            counter += 1
            project_id = f"{base_id}_{counter}"
        
        return project_id
    
    def _project_exists(self, project_id: str) -> bool:
        """Check if project exists."""
        project_path = self.projects_root / project_id
        return project_path.exists() and (project_path / "project_config.json").exists()
    
    def _create_project_structure(self, project_path: Path) -> None:
        """Create project directory structure."""
        directories = [
            "inputs",
            "stage1_outputs",
            "stage2_execution/natural_dataset",
            "stage2_execution/controlled_dataset", 
            "stage3_analysis/comparative_analysis",
            "reports"
        ]
        
        for directory in directories:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep = dir_path / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
    
    def create_project(
        self,
        brand_context: BrandContext,
        category: str,
        customer_narrative: Optional[str] = None
    ) -> 'BrandAuditProject':
        """Create new project with validation and security checks."""
        
        correlation_id = str(uuid4())
        logger.set_context(correlation_id=correlation_id, operation="create_project")
        
        try:
            # Generate project ID
            project_id = self._generate_project_id(brand_context.brand_name, category)
            project_path = self.projects_root / project_id
            
            # Security validation
            if not self.security.validate_file_path(Path(project_id), self.projects_root):
                raise SecurityError("Invalid project path")
            
            # Check if project already exists
            if self._project_exists(project_id):
                raise ProjectExistsError(f"Project {project_id} already exists")
            
            logger.info("Creating new project", 
                       metadata={
                           "project_id": project_id,
                           "brand": brand_context.brand_name,
                           "category": category
                       })
            
            # Create directory structure
            self._create_project_structure(project_path)
            
            # Create project metadata
            metadata = ProjectMetadata(
                project_id=project_id,
                display_name=f"{brand_context.brand_name} {category.title()} Brand Audit",
                brand=brand_context.brand_name,
                category=category
            )
            
            # Create project config
            project_config = ProjectConfig(project_metadata=metadata)
            
            # Save project config
            config_path = project_path / "project_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(project_config.dict(), f, indent=2, default=str)
            
            # Save brand context
            brand_path = project_path / "inputs" / "brand_context.json"
            with open(brand_path, 'w', encoding='utf-8') as f:
                json.dump(brand_context.dict(), f, indent=2)
            
            # Save customer narrative if provided
            if customer_narrative:
                narrative_path = project_path / "inputs" / "customer_narrative.txt"
                with open(narrative_path, 'w', encoding='utf-8') as f:
                    f.write(customer_narrative)
            
            # Update index
            index = self._load_index()
            index[project_id] = {
                "display_name": metadata.display_name,
                "brand": metadata.brand,
                "category": metadata.category,
                "created_date": metadata.created_date.isoformat(),
                "status": "created"
            }
            self._save_index(index)
            
            logger.info("Project created successfully", 
                       metadata={"project_id": project_id})
            
            return BrandAuditProject(project_path, self.security)
            
        except Exception as e:
            logger.error("Failed to create project", 
                        metadata={"error": str(e)}, exc_info=True)
            # Cleanup on failure
            if 'project_path' in locals() and project_path.exists():
                try:
                    shutil.rmtree(project_path)
                except OSError:
                    pass
            raise
        finally:
            logger.clear_context()
    
    def load_project(self, project_id: str) -> 'BrandAuditProject':
        """Load existing project with validation."""
        
        # Security validation
        if not self.security.validate_file_path(Path(project_id), self.projects_root):
            raise SecurityError("Invalid project path")
        
        project_path = self.projects_root / project_id
        
        if not self._project_exists(project_id):
            raise ProjectNotFoundError(f"Project {project_id} not found")
        
        logger.info("Loading project", metadata={"project_id": project_id})
        
        return BrandAuditProject(project_path, self.security)
    
    def list_projects(self) -> List[Dict]:
        """List all projects with status information."""
        
        logger.info("Listing projects")
        
        index = self._load_index()
        projects = []
        
        for project_id, metadata in index.items():
            project_path = self.projects_root / project_id
            
            if not self._project_exists(project_id):
                # Remove from index if project doesn't exist
                continue
            
            try:
                project = BrandAuditProject(project_path, self.security)
                status = project.get_status()
                
                projects.append({
                    "project_id": project_id,
                    "display_name": metadata["display_name"],
                    "brand": metadata["brand"],
                    "category": metadata["category"],
                    "created_date": metadata["created_date"],
                    "status": status
                })
                
            except Exception as e:
                logger.warning("Failed to load project status", 
                              metadata={"project_id": project_id, "error": str(e)})
                continue
        
        # Sort by creation date (newest first)
        projects.sort(key=lambda x: x["created_date"], reverse=True)
        
        logger.info("Projects listed", metadata={"project_count": len(projects)})
        
        return projects
    
    def delete_project(self, project_id: str) -> None:
        """Delete project with confirmation and backup."""
        
        # Security validation
        if not self.security.validate_file_path(Path(project_id), self.projects_root):
            raise SecurityError("Invalid project path")
        
        project_path = self.projects_root / project_id
        
        if not self._project_exists(project_id):
            raise ProjectNotFoundError(f"Project {project_id} not found")
        
        logger.info("Deleting project", metadata={"project_id": project_id})
        
        try:
            # Create backup before deletion
            backup_name = f"{project_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.projects_root / backup_name
            shutil.copytree(project_path, backup_path)
            
            # Remove project directory
            shutil.rmtree(project_path)
            
            # Update index
            index = self._load_index()
            if project_id in index:
                del index[project_id]
            self._save_index(index)
            
            logger.info("Project deleted successfully", 
                       metadata={"project_id": project_id, "backup": str(backup_path)})
                       
        except Exception as e:
            logger.error("Failed to delete project", 
                        metadata={"project_id": project_id, "error": str(e)})
            raise FileOperationError(
                f"Failed to delete project {project_id}",
                file_path=str(project_path),
                operation="project_deletion"
            ) from e


class BrandAuditProject:
    """Individual project management with security and validation."""
    
    def __init__(self, project_path: Path, security: SecurityValidator):
        self.project_path = project_path
        self.security = security
        self.config_path = project_path / "project_config.json"
        self._config: Optional[ProjectConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load project configuration with validation."""
        try:
            if not self.config_path.exists():
                raise ProjectCorruptedError("Project configuration file not found")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self._config = ProjectConfig(**config_data)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to load project config", 
                        metadata={"config_path": str(self.config_path), "error": str(e)})
            raise ProjectCorruptedError("Invalid project configuration") from e
        except OSError as e:
            logger.error("Failed to read project config file", 
                        metadata={"config_path": str(self.config_path), "error": str(e)})
            raise FileOperationError(
                "Failed to read project configuration",
                file_path=str(self.config_path),
                operation="config_load"
            ) from e
    
    def _save_config(self) -> None:
        """Save project configuration atomically."""
        if not self._config:
            return
        
        temp_file = self.config_path.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._config.dict(), f, indent=2, default=str)
            
            # Atomic replace
            temp_file.replace(self.config_path)
            
        except OSError as e:
            logger.error("Failed to save project config", 
                        metadata={"error": str(e)})
            if temp_file.exists():
                temp_file.unlink()
            raise FileOperationError(
                "Failed to save project configuration",
                file_path=str(self.config_path),
                operation="config_save"
            ) from e
    
    @property
    def project_id(self) -> str:
        """Get project ID."""
        return self._config.project_metadata.project_id
    
    @property
    def display_name(self) -> str:
        """Get project display name."""
        return self._config.project_metadata.display_name
    
    @property
    def config(self) -> ProjectConfig:
        """Get project configuration."""
        return self._config
    
    def update_stage_status(
        self, 
        stage: str, 
        status: StageStatusEnum,
        outputs: Optional[List[str]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update stage status with validation."""
        
        if stage not in self._config.stage_status:
            raise ValueError(f"Invalid stage: {stage}")
        
        stage_status = self._config.stage_status[stage]
        
        if status == StageStatusEnum.IN_PROGRESS and stage_status.status == StageStatusEnum.PENDING:
            stage_status.started_date = datetime.now()
        elif status == StageStatusEnum.COMPLETE and stage_status.status == StageStatusEnum.IN_PROGRESS:
            stage_status.completion_date = datetime.now()
            if outputs:
                stage_status.outputs = outputs
        elif status == StageStatusEnum.FAILED:
            stage_status.error_message = error_message
            stage_status.retry_count += 1
        
        stage_status.status = status
        self._config.project_metadata.last_modified = datetime.now()
        
        self._save_config()
        
        logger.info("Stage status updated", 
                   metadata={
                       "project_id": self.project_id,
                       "stage": stage,
                       "status": status.value
                   })
    
    def get_status(self) -> Dict:
        """Get comprehensive project status."""
        stages = self._config.stage_status
        
        stage1_complete = stages["stage1"].status == StageStatusEnum.COMPLETE
        stage2_complete = stages.get("stage2", StageStatus()).status == StageStatusEnum.COMPLETE
        stage3_complete = stages.get("stage3", StageStatus()).status == StageStatusEnum.COMPLETE
        
        if stage3_complete:
            completion = 100
            current_stage = "Complete"
        elif stage2_complete:
            completion = 85
            current_stage = "Stage 3 Ready"
        elif stages.get("stage2", StageStatus()).status == StageStatusEnum.IN_PROGRESS:
            completion = 60
            current_stage = "Stage 2 In Progress"
        elif stage1_complete:
            completion = 35
            current_stage = "Stage 2 Ready"
        elif stages["stage1"].status == StageStatusEnum.IN_PROGRESS:
            completion = 15
            current_stage = "Stage 1 In Progress"
        else:
            completion = 0
            current_stage = "Stage 1 Pending"
        
        return {
            "current_stage": current_stage,
            "completion_percentage": completion,
            "stage1_complete": stage1_complete,
            "stage2_started": stages.get("stage2", StageStatus()).status != StageStatusEnum.PENDING,
            "stage2_in_progress": stages.get("stage2", StageStatus()).status == StageStatusEnum.IN_PROGRESS,
            "stage2_complete": stage2_complete,
            "stage3_complete": stage3_complete,
            "has_errors": any(s.status == StageStatusEnum.FAILED for s in stages.values())
        }
    
    def get_file_path(self, relative_path: str) -> Path:
        """Get secure file path within project."""
        file_path = Path(relative_path)
        
        # Security validation
        if not self.security.validate_file_path(file_path, self.project_path):
            raise SecurityError(f"Invalid file path: {relative_path}")
        
        return self.project_path / file_path