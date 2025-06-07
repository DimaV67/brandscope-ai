"""
Interactive menu-driven CLI for Brandscope AI Brand Audit System.
"""
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

import click

from src.core.project_manager import ProjectManager, BrandAuditProject
from src.models.brand import BrandContext, ProductInfo, PriceTierEnum, CompetitiveContext
from src.models.project import StageStatusEnum
from src.utils.config import get_config
from src.utils.exceptions import (
    BrandscopeError, ProjectNotFoundError, ProjectExistsError,
    SecurityError, ValidationError
)
from src.utils.logging import get_logger, setup_logging
from src.commands.stage1 import execute_stage1
from src.commands.stage2 import execute_stage2_command, show_stage2_status, validate_stage2_prerequisites


logger = get_logger(__name__)


class BrandscopeCLI:
    """Interactive CLI application."""
    
    def __init__(self):
        self.config = get_config()
        self.project_manager = ProjectManager()
        self.current_project: Optional[BrandAuditProject] = None
        setup_logging()
        
    def run(self) -> None:
        """Run the interactive CLI."""
        try:
            self.show_welcome()
            while True:
                if self.current_project:
                    self.show_project_menu()
                else:
                    self.show_main_menu()
        except KeyboardInterrupt:
            if click.confirm("\n\n🤔 Are you sure you want to exit?", default=True):
                click.echo("👋 Thanks for using Brandscope AI!")
                sys.exit(0)
            else:
                click.echo("Continuing...")
                click.clear()
                self.run()
        except Exception as e:
            click.echo(f"\n💥 Unexpected error: {str(e)}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def show_welcome(self) -> None:
        """Show welcome message."""
        click.clear()
        click.echo("🎯 " + "=" * 60)
        click.echo("🎯 Brandscope AI Brand Audit System")
        click.echo("🎯 Interactive Mode")
        click.echo("🎯 " + "=" * 60)
        click.echo()
        click.echo("Transform brand evaluation into data-driven competitive intelligence")
        click.echo("through systematic AI behavior analysis.")
        click.echo()
        click.echo("💡 Tip: Use Ctrl+C anytime to cancel operations or exit")
        click.echo()
    
    def show_main_menu(self) -> None:
        """Show main menu when no project is open."""
        click.echo("📋 Main Menu")
        click.echo("=" * 30)
        click.echo("1. 📝 Create New Project")
        click.echo("2. 📂 Open Existing Project")
        click.echo("3. 📊 List All Projects")
        click.echo("4. ⚙️  Settings")
        click.echo("5. ❓ Help")
        click.echo("6. 🚪 Exit")
        click.echo()
        
        choice = click.prompt("Select option", type=click.IntRange(1, 6))
        
        try:
            if choice == 1:
                self.create_new_project()
            elif choice == 2:
                self.open_project_interactive()
            elif choice == 3:
                self.list_projects()
                self.pause()
            elif choice == 4:
                self.show_settings()
            elif choice == 5:
                self.show_help()
            elif choice == 6:
                self.exit_app()
        except (click.Abort, KeyboardInterrupt):
            click.echo("\nOperation cancelled.")
        except Exception as e:
            click.echo(f"\n❌ Error: {str(e)}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            self.pause()
    
    def show_project_menu(self) -> None:
        """Show project menu when a project is open."""
        project = self.current_project
        status = project.get_status()
        
        click.echo(f"📁 Project: {project.display_name}")
        click.echo(f"📊 Status: {status['current_stage']} ({status['completion_percentage']}%)")
        click.echo("=" * 50)
        
        # Show stage status
        stages = [
            ("Stage 1: Prompt Generation", status['stage1_complete']),
            ("Stage 2: AI Prompt Creation", status['stage2_complete']),
            ("Stage 3: Analysis Processing", status['stage3_complete'])
        ]
        
        for stage_name, completed in stages:
            icon = "✅" if completed else "⚪"
            click.echo(f"{icon} {stage_name}")
        click.echo()
        
        # Build dynamic menu options
        options = []
        actions = []
        
        # Stage 1 options
        if project.config.stage_status['stage1'].status == StageStatusEnum.PENDING:
            options.append("🚀 Generate Stage 1 Prompts")
            actions.append("stage1")
        else:
            options.append("📊 View Stage 1 Results")
            actions.append("view_stage1")
            options.append("🔄 Regenerate Stage 1")
            actions.append("regenerate_stage1")
        
        # Stage 2 options
        if status['stage1_complete']:
            stage2_status = project.config.stage_status.get('stage2', type('obj', (object,), {'status': StageStatusEnum.PENDING})()).status
            
            if stage2_status == StageStatusEnum.PENDING:
                options.append("🚀 Generate Stage 2 Prompts")
                actions.append("stage2")
            elif stage2_status in [StageStatusEnum.IN_PROGRESS, StageStatusEnum.COMPLETE]:
                options.append("📊 View Stage 2 Results")
                actions.append("view_stage2")
                options.append("🔄 Regenerate Stage 2")
                actions.append("regenerate_stage2")
                
                # Check if manual execution is needed
                if not status['stage2_complete']:
                    options.append("📋 View Execution Guide")
                    actions.append("view_execution_guide")
                    options.append("📤 Upload Manual Results")
                    actions.append("upload_results")
        
        # Stage 3 options (future)
        if status['stage2_complete']:
            options.append("🚀 Generate Stage 3 Analysis (Coming Soon)")
            actions.append("stage3")
        
        # Always available options
        options.extend([
            "📁 Browse Project Files", 
            "⚙️  Project Settings", 
            "🔄 Refresh Status",
            "🔙 Back to Main Menu", 
            "❓ Help", 
            "🚪 Exit"
        ])
        actions.extend(["files", "settings", "refresh", "close", "help", "exit"])
        
        # Display menu
        for i, option in enumerate(options, 1):
            click.echo(f"{i:2d}. {option}")
        click.echo()
        
        choice = click.prompt("Select option", type=click.IntRange(1, len(options)))
        action = actions[choice - 1]
        
        try:
            self.handle_project_action(action)
        except (click.Abort, KeyboardInterrupt):
            click.echo("\nOperation cancelled.")
        except Exception as e:
            click.echo(f"\n❌ Error: {str(e)}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            self.pause()

    def handle_project_action(self, action: str) -> None:
        """Handle project-specific actions."""
        if action == "stage1":
            execute_stage1(self.current_project)
        
        elif action == "regenerate_stage1":
            if click.confirm("\nThis will overwrite existing Stage 1 outputs. Are you sure?", default=False):
                click.clear()
                click.echo("📝 Provide New Inputs for Stage 1 Regeneration\n" + "=" * 40)
                brand_context, customer_narrative, category = self._get_project_inputs_from_user()

                click.echo("\n🔄 Overwriting input files...")
                brand_path = self.current_project.get_file_path("inputs/brand_context.json")
                with open(brand_path, 'w') as f:
                    f.write(brand_context.model_dump_json(indent=2))

                narrative_path = self.current_project.get_file_path("inputs/customer_narrative.txt")
                if customer_narrative:
                    with open(narrative_path, 'w') as f:
                        f.write(customer_narrative)
                elif narrative_path.exists():
                    narrative_path.unlink()
                
                click.echo("✅ Inputs updated. Starting regeneration...")
                execute_stage1(self.current_project)
            else:
                click.echo("Regeneration cancelled.")
        
        elif action == "stage2":
            asyncio.run(execute_stage2_command(self.current_project))
        
        elif action == "regenerate_stage2":
            if click.confirm("\nThis will overwrite existing Stage 2 outputs. Are you sure?", default=False):
                click.echo("🔄 Regenerating Stage 2 prompts...")
                asyncio.run(execute_stage2_command(self.current_project))
            else:
                click.echo("Regeneration cancelled.")
        
        elif action in ("view_stage1", "view_stage2", "view_execution_guide", "upload_results", 
                       "files", "settings", "refresh", "close", "help", "exit"):
            handler_map = {
                "view_stage1": self.view_stage1_results,
                "view_stage2": self.view_stage2_results,
                "view_execution_guide": self.view_execution_guide,
                "upload_results": self.upload_manual_results,
                "files": self.browse_project_files,
                "settings": self.show_project_settings,
                "refresh": self.refresh_status,
                "close": self.close_project,
                "help": self.show_project_help,
                "exit": self.exit_app
            }
            handler_map[action]()
        else:
            click.echo(f"Action '{action}' not implemented yet.")
        
        if action not in ["close", "exit"]:
            self.pause()

    def _get_project_inputs_from_user(self) -> Tuple[BrandContext, Optional[str], str]:
        """Prompts user for all project inputs and returns them."""
        brand_name = click.prompt("Brand name")
        category = click.prompt("Product category (e.g., speakers, skincare, snacks)")
        
        products = []
        click.echo("\n📦 Add Products:")
        while True:
            if products and not click.confirm(f"Add another product? (Currently have {len(products)})", default=False):
                break
            
            name = click.prompt(f"Product #{len(products) + 1} name")
            product_type = click.prompt("Product type", default=category)
            
            click.echo("\nPrice tier:")
            click.echo("1. Budget\n2. Midrange\n3. Premium")
            tier_choice = click.prompt("Select tier", type=click.IntRange(1, 3), default=2)
            price_tier = [PriceTierEnum.BUDGET, PriceTierEnum.MIDRANGE, PriceTierEnum.PREMIUM][tier_choice - 1]
            
            products.append(ProductInfo(name=name, product_type=product_type, price_tier=price_tier, key_features=[]))
            
            if len(products) >= 5:
                click.echo("Maximum products reached for demo.")
                break
        
        competitors = []
        if click.confirm("\nAdd competitor information?", default=False):
            while len(competitors) < 5:
                competitor = click.prompt(f"Competitor #{len(competitors) + 1} (or Enter to finish)", default="")
                if not competitor:
                    break
                competitors.append(competitor)
        
        brand_context = BrandContext(
            brand_name=brand_name, 
            products=products, 
            competitive_context=CompetitiveContext(primary_competitors=competitors)
        )
        
        customer_narrative = None
        if click.confirm("\nAdd customer narrative?", default=False):
            click.echo("Enter customer narrative (type '.end' on a new line to finish):")
            lines = []
            while True:
                line = sys.stdin.readline()
                if line.strip().lower() == ".end":
                    break
                lines.append(line)
            customer_narrative = "".join(lines).strip()

        return brand_context, customer_narrative, category

    def create_new_project(self) -> None:
        """Interactive project creation."""
        click.clear()
        click.echo("📝 Create New Project\n" + "=" * 30)
        
        brand_context, customer_narrative, category = self._get_project_inputs_from_user()
        
        click.echo("\n🔄 Creating project...")
        project = self.project_manager.create_project(
            brand_context=brand_context, 
            category=category, 
            customer_narrative=customer_narrative
        )
        
        click.echo(f"✅ Project created: {project.project_id}")
        self.current_project = project
        
        if click.confirm("\nGenerate Stage 1 prompts now?", default=True):
            self.handle_project_action("stage1")
        
        self.pause()
    
    def open_project_interactive(self) -> None:
        """Interactive project selection."""
        click.clear()
        projects = self.project_manager.list_projects()
        
        if not projects:
            click.echo("📂 No existing projects found.")
            if click.confirm("Create a new project?", default=True):
                self.create_new_project()
            return
        
        click.echo("📂 Available Projects\n" + "=" * 40)
        for i, project in enumerate(projects, 1):
            status = project['status']
            click.echo(f"{i:2d}. {project['display_name']}\n"
                       f"     Brand: {project['brand']} | Category: {project['category']}\n"
                       f"     Status: {status['current_stage']} ({status['completion_percentage']}%)\n"
                       f"     ID: {project['project_id']}\n")
        
        choice = click.prompt(f"Select project (1-{len(projects)} or 0 to cancel)", type=click.IntRange(0, len(projects)))
        if choice == 0: 
            return
        
        selected = projects[choice - 1]
        self.current_project = self.project_manager.load_project(selected['project_id'])
        click.echo(f"✅ Opened project: {self.current_project.display_name}")
        self.pause()
    
    def view_stage1_results(self) -> None:
        """View Stage 1 generation results."""
        stage1_path = self.current_project.project_path / "stage1_outputs"
        
        if not stage1_path.exists():
            click.echo("❌ No Stage 1 results found.")
            return
        
        click.echo("\n📊 Stage 1 Results")
        click.echo("=" * 30)
        
        json_files = list(stage1_path.glob("*.json"))
        for file_path in sorted(json_files):
            size_kb = file_path.stat().st_size / 1024
            click.echo(f"📄 {file_path.name} ({size_kb:.1f} KB)")
        
        exec_files = [f for f in json_files if "execution_package" in f.name]
        if exec_files:
            try:
                with open(exec_files[0]) as f:
                    package = json.load(f)
                
                click.echo(f"\n🎯 Execution Package:")
                click.echo(f"   Archetypes: {len(package.get('customer_archetypes', []))}")
                click.echo(f"   Queries: {len(package.get('execution_queries', []))}")
                click.echo(f"   Platforms: {len(package.get('platforms', []))}")
                
            except Exception:
                pass
    
    def view_stage2_results(self) -> None:
        """View Stage 2 generation results."""
        show_stage2_status(self.current_project)
    
    def view_execution_guide(self) -> None:
        """View Stage 2 execution guide."""
        stage2_path = self.current_project.project_path / "stage2_execution"
        
        if not stage2_path.exists():
            click.echo("❌ Stage 2 execution directory not found.")
            return
        
        guide_files = list(stage2_path.glob("*guide*.md"))
        
        if not guide_files:
            click.echo("❌ Execution guide not found. Generate Stage 2 first.")
            return
        
        guide_path = sorted(guide_files)[-1]  # Get latest guide
        
        try:
            with open(guide_path, 'r') as f:
                content = f.read()
            
            click.echo("\n📋 Execution Guide Preview\n" + "=" * 40)
            lines = content.split('\n')
            
            # Show first 30 lines
            for line in lines[:30]:
                click.echo(line)
            
            if len(lines) > 30:
                click.echo(f"\n... ({len(lines) - 30} more lines)")
                if click.confirm("Show full guide?", default=False):
                    for line in lines[30:]:
                        click.echo(line)
            
            click.echo(f"\n📁 Full guide location: {guide_path}")
            
        except Exception as e:
            click.echo(f"❌ Error reading guide: {e}")
    
    def upload_manual_results(self) -> None:
        """Upload manual execution results for analysis."""
        stage2_path = self.current_project.project_path / "stage2_execution"
        
        if not stage2_path.exists():
            click.echo("❌ Stage 2 execution directory not found.")
            return
        
        # Check for response files
        natural_dir = stage2_path / "natural_dataset"
        controlled_dir = stage2_path / "controlled_dataset"
        
        natural_responses = list(natural_dir.glob("*.json")) if natural_dir.exists() else []
        controlled_responses = list(controlled_dir.glob("*.json")) if controlled_dir.exists() else []
        
        click.echo(f"\n📤 Manual Results Upload Status")
        click.echo("=" * 40)
        click.echo(f"Natural responses found: {len(natural_responses)}")
        click.echo(f"Controlled responses found: {len(controlled_responses)}")
        
        if not natural_responses and not controlled_responses:
            click.echo("❌ No response files found.")
            click.echo("💡 Complete manual execution first following the execution guide.")
            return
        
        # Validate file structure
        click.echo("\n🔍 Validating file structure...")
        
        valid_files = 0
        total_files = len(natural_responses) + len(controlled_responses)
        
        for file_path in natural_responses + controlled_responses:
            try:
                with open(file_path, 'r') as f:
                    json.load(f)  # Validate JSON
                valid_files += 1
            except json.JSONDecodeError:
                click.echo(f"❌ Invalid JSON: {file_path.name}")
        
        click.echo(f"✅ Valid files: {valid_files}/{total_files}")
        
        if valid_files > 0:
            # Update project status to indicate manual execution complete
            self.current_project.update_stage_status("stage2", StageStatusEnum.COMPLETE)
            click.echo("✅ Manual execution results validated and uploaded")
            click.echo("🚀 Ready to proceed to Stage 3 analysis")
        else:
            click.echo("❌ No valid response files found")
    
    def browse_project_files(self) -> None:
        """Browse project file structure."""
        click.echo(f"\n📁 Project Files - {self.current_project.display_name}\n" + "=" * 50)
        
        def show_directory(path: Path, prefix: str = "", max_depth: int = 3) -> None:
            if max_depth <= 0: 
                return
            try:
                items = sorted(list(path.iterdir()), key=lambda x: (x.is_file(), x.name))
                for item in items:
                    if item.name.startswith('.'): 
                        continue
                    if item.is_dir():
                        click.echo(f"{prefix}📁 {item.name}/")
                        if any(item.iterdir()):
                            show_directory(item, prefix + "  ", max_depth - 1)
                    else:
                        size_kb = item.stat().st_size / 1024
                        click.echo(f"{prefix}📄 {item.name} ({size_kb:.1f} KB)")
            except OSError: 
                pass
        
        show_directory(self.current_project.project_path)
    
    def show_project_settings(self) -> None:
        """Show project settings and info."""
        config = self.current_project.config
        metadata = config.project_metadata
        
        click.echo(f"\n⚙️  Project Settings\n" + "=" * 30)
        click.echo(f"Name: {metadata.display_name}")
        click.echo(f"ID: {metadata.project_id}")
        click.echo(f"Brand: {metadata.brand}")
        click.echo(f"Category: {metadata.category}")
        click.echo(f"Created: {metadata.created_date.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"Modified: {metadata.last_modified.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"Version: {metadata.version}")
        
        click.echo("\nStage Status:")
        for stage, status in config.stage_status.items():
            click.echo(f"  {stage}: {status.status.value}")
            if status.error_message:
                click.echo(f"    Error: {status.error_message}")
    
    def list_projects(self) -> None:
        """List all projects with status."""
        projects = self.project_manager.list_projects()
        if not projects:
            click.echo("📂 No projects found.")
            return
        
        click.echo(f"\n📂 All Projects ({len(projects)})\n" + "=" * 50)
        for i, project in enumerate(projects, 1):
            status = project['status']
            click.echo(f"{i:2d}. {project['display_name']}\n"
                       f"     Brand: {project['brand']} | Category: {project['category']}\n"
                       f"     Status: {status['current_stage']} ({status['completion_percentage']}%)\n"
                       f"     Created: {project['created_date'][:10]}\n")
    
    def show_settings(self) -> None:
        """Show application settings."""
        click.echo("\n⚙️  Application Settings\n" + "=" * 30)
        click.echo(f"Debug Mode: {self.config.debug}")
        click.echo(f"Environment: {self.config.environment}")
        click.echo(f"Log Level: {self.config.log_level}")
        click.echo(f"Projects Root: {self.config.projects_root}")
        click.echo(f"Cache Root: {self.config.cache_root}")
        self.pause()
    
    def show_help(self) -> None:
        """Show help information."""
        click.echo("\n❓ Brandscope AI Help\n" + "=" * 25)
        click.echo("🎯 Purpose: Generate AI brand audit intelligence")
        click.echo("📋 Workflow:\n"
                   "   1. Create project with brand/product info\n"
                   "   2. Generate Stage 1 prompts and archetypes\n"
                   "   3. Generate Stage 2 AI prompts (natural & controlled)\n"
                   "   4. Execute manual AI testing across platforms\n"
                   "   5. Upload results for analysis (Stage 3)")
        click.echo("\n💡 Tips:\n"
                   "   - Use clear, specific brand/product names\n"
                   "   - Add competitor info for better analysis\n"
                   "   - Customer narratives improve accuracy\n"
                   "   - Stage 1 generates customer archetypes\n"
                   "   - Stage 2 creates platform-ready prompts")
        self.pause()
    
    def show_project_help(self) -> None:
        """Show project-specific help."""
        status = self.current_project.get_status()
        click.echo("\n❓ Project Help\n" + "=" * 20)
        
        if not status['stage1_complete']:
            click.echo("🚀 Next: Generate Stage 1 prompts\n"
                       "   This creates customer archetypes and test queries")
        elif not status['stage2_started']:
            click.echo("🚀 Next: Generate Stage 2 prompts\n"
                       "   This creates natural and controlled AI prompts")
        elif status['stage2_in_progress']:
            click.echo("📋 Next: Complete manual execution\n"
                       "   Follow execution guide to test prompts on AI platforms")
        elif status['stage2_complete']:
            click.echo("📤 Next: Upload manual results or generate analysis\n"
                       "   Upload response files or proceed to Stage 3")
        else:
            click.echo("✅ Project complete!")
        
        click.echo("\n🔧 Available Actions:")
        click.echo("   • View results: See generated files and status")
        click.echo("   • Regenerate: Re-run any stage with new inputs")
        click.echo("   • Browse files: Explore project directory")
        click.echo("   • Settings: View project configuration")
        
        self.pause()
    
    def refresh_status(self):
        """Reloads the project config and shows a message."""
        self.current_project._load_config()
        click.echo("✅ Status refreshed.")

    def close_project(self):
        """Closes the current project."""
        click.echo(f"📁 Closed project: {self.current_project.display_name}")
        self.current_project = None

    def exit_app(self):
        """Exits the application."""
        click.echo("\n👋 Thanks for using Brandscope AI!")
        sys.exit(0)
    
    def pause(self) -> None:
        """Pause for user input."""
        click.echo()
        click.prompt("Press Enter to continue", default="", show_default=False)
        click.clear()


@click.command()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug: bool) -> None:
    """🎯 Brandscope AI Brand Audit System - Interactive Mode"""
    from src.cli import BrandscopeCLI
    app = BrandscopeCLI()
    if debug:
        app.config.debug = True
        app.config.log_level = "DEBUG"
    app.run()


if __name__ == '__main__':
    cli()