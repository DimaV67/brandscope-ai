"""
Interactive menu-driven CLI for Brandscope AI Brand Audit System.
"""
import sys
from pathlib import Path
from typing import List, Optional

import click

from .core.project_manager import ProjectManager, BrandAuditProject
from .models.brand import BrandContext, ProductInfo, PriceTierEnum, CompetitiveContext
from .models.project import StageStatusEnum
from .utils.config import get_config
from .utils.exceptions import (
    BrandscopeError, ProjectNotFoundError, ProjectExistsError,
    SecurityError, ValidationError
)
from .utils.logging import get_logger, setup_logging
from .commands.stage1 import execute_stage1


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
            # Handle Ctrl+C gracefully
            if click.confirm("\n\n🤔 Are you sure you want to exit?", default=True):
                click.echo("👋 Thanks for using Brandscope AI!")
                sys.exit(0)
            else:
                click.echo("Continuing...")
                click.clear()
                self.run()  # Resume
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
                click.echo("\n👋 Thanks for using Brandscope AI!")
                sys.exit(0)
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
        
        # Show stage indicators
        stages = [
            ("Stage 1: Prompt Generation", status['stage1_complete']),
            ("Stage 2: Manual Execution", status['stage2_complete']),
            ("Stage 3: Analysis Processing", status['stage3_complete'])
        ]
        
        for stage_name, completed in stages:
            icon = "✅" if completed else "⚪"
            click.echo(f"{icon} {stage_name}")
        click.echo()
        
        # Dynamic menu based on project status
        options = []
        actions = []
        
        if not status['stage1_complete']:
            options.append("🚀 Generate Stage 1 Prompts")
            actions.append("stage1")
        else:
            options.append("📊 View Stage 1 Results")
            actions.append("view_stage1")
            
            if not status['stage2_started']:
                options.append("▶️  Start Stage 2 (Manual Execution)")
                actions.append("start_stage2")
            elif status['stage2_in_progress']:
                options.append("📋 View Stage 2 Guide")
                actions.append("view_guide")
                options.append("📤 Upload Stage 2 Results")
                actions.append("upload_stage2")
            elif status['stage2_complete']:
                options.append("📊 View Stage 2 Results")
                actions.append("view_stage2")
                if not status['stage3_complete']:
                    options.append("🚀 Generate Stage 3 Analysis")
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
    
    def create_new_project(self) -> None:
        """Interactive project creation."""
        click.clear()
        click.echo("📝 Create New Project")
        click.echo("=" * 30)
        
        # Get brand information
        brand_name = click.prompt("Brand name")
        category = click.prompt("Product category (e.g., speakers, skincare, snacks)")
        
        # Get products
        products = []
        click.echo("\n📦 Add Products:")
        while True:
            if products:
                if not click.confirm(f"Add another product? (Currently have {len(products)})", default=False):
                    break
            
            name = click.prompt(f"Product #{len(products) + 1} name")
            product_type = click.prompt("Product type", default=category)
            
            # Simple price tier selection
            click.echo("\nPrice tier:")
            click.echo("1. Budget")
            click.echo("2. Midrange") 
            click.echo("3. Premium")
            tier_choice = click.prompt("Select tier", type=click.IntRange(1, 3), default=2)
            price_tier = [PriceTierEnum.BUDGET, PriceTierEnum.MIDRANGE, PriceTierEnum.PREMIUM][tier_choice - 1]
            
            products.append(ProductInfo(
                name=name,
                product_type=product_type,
                price_tier=price_tier,
                key_features=[]
            ))
            
            if len(products) >= 5:
                click.echo("Maximum products reached for demo.")
                break
        
        # Optional competitive context
        competitors = []
        if click.confirm("\nAdd competitor information?", default=False):
            while len(competitors) < 5:
                competitor = click.prompt(f"Competitor #{len(competitors) + 1} (or Enter to finish)", default="")
                if not competitor:
                    break
                competitors.append(competitor)
        
        # Create brand context
        brand_context = BrandContext(
            brand_name=brand_name,
            products=products,
            competitive_context=CompetitiveContext(primary_competitors=competitors)
        )
        
        # Optional customer narrative
        customer_narrative = None
        if click.confirm("\nAdd customer narrative?", default=False):
            click.echo("Enter customer narrative (Ctrl+C when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except KeyboardInterrupt:
                customer_narrative = '\n'.join(lines).strip()
        
        # Create project
        click.echo("\n🔄 Creating project...")
        project = self.project_manager.create_project(
            brand_context=brand_context,
            category=category,
            customer_narrative=customer_narrative
        )
        
        click.echo(f"✅ Project created: {project.project_id}")
        self.current_project = project
        
        # Offer immediate Stage 1 generation
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
            return
        
        click.echo("📂 Available Projects")
        click.echo("=" * 40)
        
        for i, project in enumerate(projects, 1):
            status = project['status']
            click.echo(f"{i:2d}. {project['display_name']}")
            click.echo(f"     Brand: {project['brand']} | Category: {project['category']}")
            click.echo(f"     Status: {status['current_stage']} ({status['completion_percentage']}%)")
            click.echo(f"     ID: {project['project_id']}")
            click.echo()
        
        choice = click.prompt(f"Select project (1-{len(projects)} or 0 to cancel)", 
                             type=click.IntRange(0, len(projects)))
        
        if choice == 0:
            return
        
        selected = projects[choice - 1]
        self.current_project = self.project_manager.load_project(selected['project_id'])
        click.echo(f"✅ Opened project: {self.current_project.display_name}")
        self.pause()
    
    def handle_project_action(self, action: str) -> None:
        """Handle project-specific actions."""
        if action == "stage1":
            click.echo("\n🚀 Generating Stage 1 Prompts...")
            execute_stage1(self.current_project)
            
        elif action == "view_stage1":
            self.view_stage1_results()
            
        elif action == "start_stage2":
            self.start_stage2()
            
        elif action == "view_guide":
            self.view_execution_guide()
            
        elif action == "files":
            self.browse_project_files()
            
        elif action == "settings":
            self.show_project_settings()
            
        elif action == "refresh":
            # Just refresh by reloading project
            self.current_project._load_config()
            click.echo("✅ Status refreshed.")
            
        elif action == "close":
            click.echo(f"📁 Closed project: {self.current_project.display_name}")
            self.current_project = None
            
        elif action == "help":
            self.show_project_help()
        
        elif action == "exit":
            click.echo("\n👋 Thanks for using Brandscope AI!")
            sys.exit(0)
            
        else:
            click.echo(f"Action '{action}' not implemented yet.")
        
        if action != "close":
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
        
        # Show execution package summary
        exec_files = [f for f in json_files if "execution_package" in f.name]
        if exec_files:
            try:
                import json
                with open(exec_files[0]) as f:
                    package = json.load(f)
                
                click.echo(f"\n🎯 Execution Package:")
                click.echo(f"   Archetypes: {len(package.get('customer_archetypes', []))}")
                click.echo(f"   Queries: {len(package.get('execution_queries', []))}")
                click.echo(f"   Platforms: {len(package.get('platforms', []))}")
                
            except Exception:
                pass
    
    def start_stage2(self) -> None:
        """Start Stage 2 manual execution."""
        guide_path = self.current_project.project_path / "stage2_execution" / "manual_execution_guide.md"
        
        if not guide_path.exists():
            click.echo("❌ Execution guide not found. Run Stage 1 first.")
            return
        
        # Update status
        self.current_project.update_stage_status("stage2", StageStatusEnum.IN_PROGRESS)
        
        click.echo("\n▶️  Stage 2 Started")
        click.echo("=" * 25)
        click.echo("✅ Status updated to 'In Progress'")
        click.echo(f"📋 Execution guide: {guide_path.name}")
        click.echo(f"📁 Save responses to: stage2_execution/")
        click.echo("\n💡 Use 'View Stage 2 Guide' to see detailed instructions.")
    
    def view_execution_guide(self) -> None:
        """View Stage 2 execution guide."""
        guide_path = self.current_project.project_path / "stage2_execution" / "manual_execution_guide.md"
        
        if not guide_path.exists():
            click.echo("❌ Execution guide not found.")
            return
        
        try:
            with open(guide_path, 'r') as f:
                content = f.read()
            
            # Show preview
            lines = content.split('\n')
            click.echo("\n📋 Execution Guide Preview")
            click.echo("=" * 40)
            
            for line in lines[:25]:  # First 25 lines
                click.echo(line)
            
            if len(lines) > 25:
                click.echo(f"\n... ({len(lines) - 25} more lines)")
                if click.confirm("Show full guide?", default=False):
                    for line in lines[25:]:
                        click.echo(line)
            
            click.echo(f"\n📁 Full guide location: {guide_path}")
            
        except Exception as e:
            click.echo(f"❌ Error reading guide: {e}")
    
    def browse_project_files(self) -> None:
        """Browse project file structure."""
        click.echo(f"\n📁 Project Files - {self.current_project.display_name}")
        click.echo("=" * 50)
        
        def show_directory(path: Path, prefix: str = "", max_depth: int = 3) -> None:
            if max_depth <= 0:
                return
                
            try:
                items = list(path.iterdir())
                items.sort(key=lambda x: (x.is_file(), x.name))
                
                for item in items:
                    if item.name.startswith('.'):
                        continue
                        
                    if item.is_dir():
                        click.echo(f"{prefix}📁 {item.name}/")
                        if len(list(item.iterdir())) > 0:
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
        
        click.echo(f"\n⚙️  Project Settings")
        click.echo("=" * 30)
        click.echo(f"Name: {metadata.display_name}")
        click.echo(f"ID: {metadata.project_id}")
        click.echo(f"Brand: {metadata.brand}")
        click.echo(f"Category: {metadata.category}")
        click.echo(f"Created: {metadata.created_date.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"Modified: {metadata.last_modified.strftime('%Y-%m-%d %H:%M')}")
        click.echo(f"Version: {metadata.version}")
        
        click.echo(f"\nStage Status:")
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
        
        click.echo(f"\n📂 All Projects ({len(projects)})")
        click.echo("=" * 50)
        
        for i, project in enumerate(projects, 1):
            status = project['status']
            click.echo(f"{i:2d}. {project['display_name']}")
            click.echo(f"     Brand: {project['brand']} | Category: {project['category']}")
            click.echo(f"     Status: {status['current_stage']} ({status['completion_percentage']}%)")
            click.echo(f"     Created: {project['created_date'][:10]}")
            click.echo()
    
    def show_settings(self) -> None:
        """Show application settings."""
        click.echo("\n⚙️  Application Settings")
        click.echo("=" * 30)
        click.echo(f"Debug Mode: {self.config.debug}")
        click.echo(f"Environment: {self.config.environment}")
        click.echo(f"Log Level: {self.config.log_level}")
        click.echo(f"Projects Root: {self.config.projects_root}")
        click.echo(f"Cache Root: {self.config.cache_root}")
        
        self.pause()
    
    def show_help(self) -> None:
        """Show help information."""
        click.echo("\n❓ Brandscope AI Help")
        click.echo("=" * 25)
        click.echo("🎯 Purpose: Generate AI brand audit intelligence")
        click.echo("📋 Workflow:")
        click.echo("   1. Create project with brand/product info")
        click.echo("   2. Generate Stage 1 prompts and archetypes")
        click.echo("   3. Execute manual AI testing (Stage 2)")
        click.echo("   4. Upload results for analysis (Stage 3)")
        click.echo()
        click.echo("💡 Tips:")
        click.echo("   - Use clear, specific brand/product names")
        click.echo("   - Add competitor info for better analysis")
        click.echo("   - Customer narratives improve accuracy")
        click.echo("   - Stage 1 generates 15-20 test queries")
        
        self.pause()
    
    def show_project_help(self) -> None:
        """Show project-specific help."""
        status = self.current_project.get_status()
        
        click.echo("\n❓ Project Help")
        click.echo("=" * 20)
        
        if not status['stage1_complete']:
            click.echo("🚀 Next: Generate Stage 1 prompts")
            click.echo("   This creates customer archetypes and test queries")
        elif not status['stage2_started']:
            click.echo("▶️  Next: Start Stage 2 manual execution")
            click.echo("   This begins the AI testing process")
        elif status['stage2_in_progress']:
            click.echo("📋 Next: Follow the execution guide")
            click.echo("   Test queries on AI platforms and save responses")
        elif status['stage2_complete']:
            click.echo("🚀 Next: Generate Stage 3 analysis")
            click.echo("   This analyzes results and creates insights")
        else:
            click.echo("✅ Project complete!")
        
        self.pause()
    
    def pause(self) -> None:
        """Pause for user input."""
        click.echo()
        click.prompt("Press Enter to continue", default="", show_default=False)
        click.clear()


# CLI entry point
@click.command()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug: bool) -> None:
    """🎯 Brandscope AI Brand Audit System - Interactive Mode"""
    app = BrandscopeCLI()
    if debug:
        app.config.debug = True
        app.config.log_level = "DEBUG"
    app.run()


if __name__ == '__main__':
    cli()