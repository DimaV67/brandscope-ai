"""
Command-line interface with rich user experience and error handling.
"""
import sys
from pathlib import Path
from typing import List, Optional

import click

from .core.project_manager import ProjectManager, BrandAuditProject
from .models.brand import BrandContext, ProductInfo, PriceTierEnum
from .utils.config import get_config
from .utils.exceptions import (
   BrandscopeError, ProjectNotFoundError, ProjectExistsError,
   SecurityError, ValidationError
)
from .utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class CLIContext:
   """CLI context for sharing state between commands."""
   
   def __init__(self):
       self.config = get_config()
       self.project_manager = ProjectManager()
       self.current_project: Optional[BrandAuditProject] = None


# Global CLI context
cli_context = CLIContext()


def handle_error(func):
   """Decorator for consistent error handling."""
   def wrapper(*args, **kwargs):
       try:
           return func(*args, **kwargs)
       except BrandscopeError as e:
           logger.error(f"Application error: {e.message}", 
                       metadata=e.details, correlation_id=e.correlation_id)
           click.echo(f"❌ Error: {e.message}", err=True)
           if cli_context.config.debug:
               click.echo(f"Details: {e.details}", err=True)
               click.echo(f"Correlation ID: {e.correlation_id}", err=True)
           sys.exit(1)
       except KeyboardInterrupt:
           click.echo("\n⚠️  Operation cancelled by user", err=True)
           sys.exit(130)
       except Exception as e:
           logger.critical("Unexpected error", exc_info=True)
           click.echo(f"💥 Unexpected error: {str(e)}", err=True)
           if cli_context.config.debug:
               import traceback
               click.echo(traceback.format_exc(), err=True)
           sys.exit(1)
   return wrapper


def validate_input(value: str, field_name: str, max_length: int = 100) -> str:
   """Validate user input with security checks."""
   if not value or not value.strip():
       raise ValidationError(f"{field_name} cannot be empty")
   
   value = value.strip()
   
   if len(value) > max_length:
       raise ValidationError(f"{field_name} cannot exceed {max_length} characters")
   
   # Basic security validation
   dangerous_chars = ['<', '>', '&', '"', "'", '`']
   if any(char in value for char in dangerous_chars):
       raise ValidationError(f"{field_name} contains invalid characters")
   
   return value


def prompt_for_products() -> List[ProductInfo]:
   """Interactive prompt for product information."""
   products = []
   
   click.echo("\n📦 Product Information")
   click.echo("Enter product details (press Enter with empty name to finish)")
   
   while True:
       click.echo(f"\n--- Product #{len(products) + 1} ---")
       
       name = click.prompt("Product name (or Enter to finish)", default="", show_default=False)
       if not name.strip():
           break
           
       name = validate_input(name, "Product name")
       
       product_type = click.prompt("Product type", default="")
       product_type = validate_input(product_type, "Product type")
       
       # Price tier selection
       click.echo("\nPrice tier options:")
       for i, tier in enumerate(PriceTierEnum, 1):
           click.echo(f"  {i}. {tier.value.title()}")
       
       tier_choice = click.prompt("Select price tier", type=click.IntRange(1, len(PriceTierEnum)))
       price_tier = list(PriceTierEnum)[tier_choice - 1]
       
       # Optional price range
       price_range = click.prompt("Price range (optional)", default="", show_default=False)
       if price_range:
           price_range = validate_input(price_range, "Price range", 50)
       else:
           price_range = None
       
       # Key features
       features = []
       click.echo("\nKey features (Enter empty line to finish):")
       while len(features) < 5:  # Limit features
           feature = click.prompt(f"Feature #{len(features) + 1}", default="", show_default=False)
           if not feature.strip():
               break
           features.append(validate_input(feature, "Feature", 200))
       
       products.append(ProductInfo(
           name=name,
           product_type=product_type,
           price_tier=price_tier,
           price_range=price_range,
           key_features=features
       ))
       
       if len(products) >= 10:  # Reasonable limit
           click.echo("⚠️  Maximum number of products reached (10)")
           break
       
       continue_adding = click.confirm("Add another product?", default=False)
       if not continue_adding:
           break
   
   if not products:
       raise ValidationError("At least one product is required")
   
   return products


def prompt_for_brand_context() -> BrandContext:
   """Interactive prompt for brand context."""
   click.echo("\n🏢 Brand Information")
   
   brand_name = click.prompt("Brand name")
   brand_name = validate_input(brand_name, "Brand name")
   
   products = prompt_for_products()
   
   # Optional competitive context
   click.echo("\n🥊 Competitive Context (Optional)")
   competitors = []
   
   add_competitors = click.confirm("Add competitor information?", default=False)
   if add_competitors:
       click.echo("Enter competitor names (press Enter with empty name to finish):")
       while len(competitors) < 10:
           competitor = click.prompt(f"Competitor #{len(competitors) + 1}", default="", show_default=False)
           if not competitor.strip():
               break
           competitors.append(validate_input(competitor, "Competitor name"))
   
   # Optional positioning and target markets
   positioning = click.prompt("Brand positioning (optional)", default="", show_default=False)
   if positioning:
       positioning = validate_input(positioning, "Brand positioning", 500)
   else:
       positioning = None
   
   target_markets = []
   add_markets = click.confirm("Add target market information?", default=False)
   if add_markets:
       click.echo("Enter target markets (press Enter with empty market to finish):")
       while len(target_markets) < 5:
           market = click.prompt(f"Target market #{len(target_markets) + 1}", default="", show_default=False)
           if not market.strip():
               break
           target_markets.append(validate_input(market, "Target market"))
   
   from .models.brand import CompetitiveContext
   competitive_context = CompetitiveContext(
       primary_competitors=competitors
   )
   
   return BrandContext(
       brand_name=brand_name,
       products=products,
       competitive_context=competitive_context,
       brand_positioning=positioning,
       target_markets=target_markets
   )


def display_project_status(project: BrandAuditProject) -> None:
   """Display project status information."""
   status = project.get_status()
   
   click.echo(f"\n🎯 Project: {project.display_name}")
   click.echo(f"📊 Status: {status['current_stage']} - {status['completion_percentage']}%")
   click.echo("=" * 50)
   
   # Stage status indicators
   stages = [
       ("Stage 1: Prompt Generation", status['stage1_complete']),
       ("Stage 2: Manual Execution", status['stage2_complete']),
       ("Stage 3: Analysis Processing", status['stage3_complete'])
   ]
   
   for stage_name, completed in stages:
       icon = "✅" if completed else "⏳" if stage_name == status['current_stage'] else "⚪"
       click.echo(f"{icon} {stage_name}")
   
   if status['has_errors']:
       click.echo("\n⚠️  Some stages have errors. Check logs for details.")


def display_projects_list(projects: List[dict]) -> None:
   """Display formatted list of projects."""
   if not projects:
       click.echo("📂 No projects found.")
       return
   
   click.echo(f"\n📂 Found {len(projects)} project(s):")
   click.echo("=" * 80)
   
   for i, project in enumerate(projects, 1):
       status_info = project['status']
       status_text = status_info['current_stage']
       completion = status_info['completion_percentage']
       
       click.echo(f"{i:2d}. {project['display_name']}")
       click.echo(f"    Brand: {project['brand']} | Category: {project['category']}")
       click.echo(f"    Status: {status_text} ({completion}%)")
       click.echo(f"    Created: {project['created_date'][:10]}")
       click.echo()


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.version_option(version='1.0.0')
@handle_error
def cli(debug: bool, config: Optional[str]) -> None:
   """🎯 Brandscope AI Brand Audit System
   
   Transform brand evaluation into data-driven competitive intelligence
   through systematic AI behavior analysis.
   """
   # Setup logging
   setup_logging()
   
   # Update config if needed
   if debug:
       cli_context.config.debug = True
       cli_context.config.log_level = "DEBUG"
   
   if config:
       from .utils.config import reload_config
       cli_context.config = reload_config(Path(config))
   
   logger.info("CLI started", metadata={
       "debug": debug,
       "config_file": config
   })


@cli.command()
@click.option('--brand', help='Brand name')
@click.option('--category', help='Product category')
@click.option('--products', help='Product names (comma-separated)')
@click.option('--narrative', type=click.File('r'), help='Customer narrative file')
@handle_error
def new(brand: Optional[str], category: Optional[str], 
       products: Optional[str], narrative: Optional[click.File]) -> None:
   """Create a new brand audit project."""
   
   click.echo("🎯 Brandscope AI Brand Audit System")
   click.echo("=" * 40)
   click.echo("📝 Creating New Brand Audit Project")
   
   # Interactive mode if parameters not provided
   if not all([brand, category]):
       if brand:
           brand = validate_input(brand, "Brand name")
           category = click.prompt("Product category")
           category = validate_input(category, "Product category")
       else:
           brand_context = prompt_for_brand_context()
           category = click.prompt("Product category")
           category = validate_input(category, "Product category")
   else:
       # Command-line mode
       brand = validate_input(brand, "Brand name")
       category = validate_input(category, "Product category")
       
       if products:
           product_names = [p.strip() for p in products.split(',')]
           product_list = []
           for name in product_names:
               if name:
                   product_list.append(ProductInfo(
                       name=validate_input(name, "Product name"),
                       product_type="unspecified",
                       price_tier=PriceTierEnum.MIDRANGE
                   ))
           
           brand_context = BrandContext(
               brand_name=brand,
               products=product_list
           )
       else:
           brand_context = prompt_for_brand_context()
   
   # Customer narrative
   customer_narrative = None
   if narrative:
       customer_narrative = narrative.read().strip()
       if customer_narrative:
           customer_narrative = validate_input(customer_narrative, "Customer narrative", 2000)
   else:
       has_narrative = click.confirm("\nDo you have a specific customer narrative?", default=False)
       if has_narrative:
           click.echo("Enter customer narrative (press Ctrl+D when finished):")
           try:
               lines = []
               while True:
                   line = input()
                   lines.append(line)
           except EOFError:
               customer_narrative = '\n'.join(lines).strip()
               if customer_narrative:
                   customer_narrative = validate_input(customer_narrative, "Customer narrative", 2000)
   
   # Create project
   try:
       project = cli_context.project_manager.create_project(
           brand_context=brand_context,
           category=category,
           customer_narrative=customer_narrative
       )
       
       click.echo(f"\n✅ Project created successfully!")
       click.echo(f"📁 Project ID: {project.project_id}")
       click.echo(f"📂 Directory: {project.project_path}")
       
       # Offer to generate Stage 1 immediately
       if click.confirm("\nGenerate Stage 1 prompts now?", default=True):
           cli_context.current_project = project
           from .commands.stage1 import execute_stage1
           execute_stage1()
       else:
           click.echo(f"\n💡 To continue later:")
           click.echo(f"   brandscope open {project.project_id}")
           click.echo(f"   brandscope stage1 --project {project.project_id}")
   
   except Exception as e:
       logger.error("Project creation failed", exc_info=True)
       raise


@cli.command()
@click.argument('project_id', required=False)
@handle_error
def open(project_id: Optional[str]) -> None:
   """Open an existing project."""
   
   if not project_id:
       # Interactive project selection
       projects = cli_context.project_manager.list_projects()
       
       if not projects:
           click.echo("❌ No existing projects found.")
           if click.confirm("Create a new project?", default=True):
               from click.testing import CliRunner
               runner = CliRunner()
               runner.invoke(new)
           return
       
       display_projects_list(projects)
       
       try:
           choice = click.prompt(f"\nSelect project (1-{len(projects)})", type=click.IntRange(1, len(projects)))
           project_id = projects[choice - 1]['project_id']
       except click.Abort:
           click.echo("Operation cancelled.")
           return
   
   # Load project
   try:
       project = cli_context.project_manager.load_project(project_id)
       cli_context.current_project = project
       
       display_project_status(project)
       
       # Show available actions
       show_project_menu(project)
       
   except ProjectNotFoundError:
       click.echo(f"❌ Project '{project_id}' not found.")
       
       # Suggest similar project IDs
       projects = cli_context.project_manager.list_projects()
       similar = [p for p in projects if project_id.lower() in p['project_id'].lower()]
       
       if similar:
           click.echo("\nDid you mean one of these?")
           for p in similar[:3]:
               click.echo(f"  - {p['project_id']}")


@cli.command()
@handle_error
def list() -> None:
   """List all projects."""
   
   projects = cli_context.project_manager.list_projects()
   display_projects_list(projects)
   
   if projects:
       click.echo("💡 To open a project: brandscope open <project_id>")


def show_project_menu(project: BrandAuditProject) -> None:
   """Show interactive project menu."""
   
   status = project.get_status()
   
   # Determine available actions
   actions = []
   
   if not status['stage1_complete']:
       actions.append(("Generate Stage 1 Prompts", "stage1"))
   else:
       actions.append(("View Stage 1 Results", "view_stage1"))
       
       if not status['stage2_started']:
           actions.append(("Start Stage 2 (Manual Execution)", "stage2_start"))
       elif status['stage2_in_progress']:
           actions.append(("Continue Stage 2", "stage2_continue"))
           actions.append(("Upload Stage 2 Results", "stage2_upload"))
       elif status['stage2_complete']:
           actions.append(("View Stage 2 Results", "view_stage2"))
           if not status['stage3_complete']:
               actions.append(("Generate Stage 3 Analysis", "stage3"))
   
   actions.extend([
       ("View Project Files", "files"),
       ("Export Results", "export"),
       ("Project Settings", "settings"),
       ("Back to Main Menu", "back")
   ])
   
   while True:
       click.echo(f"\n📋 Available Actions:")
       for i, (action_name, _) in enumerate(actions, 1):
           click.echo(f"  {i}. {action_name}")
       
       try:
           choice = click.prompt(f"\nSelect action (1-{len(actions)})", 
                               type=click.IntRange(1, len(actions)))
           action_code = actions[choice - 1][1]
           
           if action_code == "back":
               break
           elif action_code == "stage1":
               from .commands.stage1 import execute_stage1
               execute_stage1()
           elif action_code == "view_stage1":
               view_stage1_results(project)
           elif action_code == "stage2_start":
               start_stage2_execution(project)
           elif action_code == "files":
               view_project_files(project)
           elif action_code == "export":
               export_project_results(project)
           elif action_code == "settings":
               show_project_settings(project)
           else:
               click.echo(f"Action '{action_code}' not implemented yet.")
               
       except click.Abort:
           break


def view_stage1_results(project: BrandAuditProject) -> None:
   """View Stage 1 results."""
   stage1_path = project.project_path / "stage1_outputs"
   
   if not stage1_path.exists():
       click.echo("❌ No Stage 1 results found.")
       return
   
   click.echo(f"\n📊 Stage 1 Results - {project.display_name}")
   click.echo("=" * 50)
   
   # List output files
   output_files = list(stage1_path.glob("*.json"))
   if output_files:
       click.echo("Generated files:")
       for file_path in sorted(output_files):
           size_kb = file_path.stat().st_size / 1024
           click.echo(f"  📄 {file_path.name} ({size_kb:.1f} KB)")
   
   # Show execution package summary if available
   exec_package_path = stage1_path / "execution_package.json"
   if exec_package_path.exists():
       try:
           import json
           with open(exec_package_path) as f:
               package = json.load(f)
           
           click.echo(f"\n🎯 Ready for execution:")
           click.echo(f"  Customer archetypes: {len(package.get('customer_archetypes', []))}")
           click.echo(f"  Styled queries: {len(package.get('styled_queries', []))}")
           click.echo(f"  Execution priority: {package.get('execution_priority', 'Not set')}")
           
       except (json.JSONDecodeError, OSError):
           pass
   
   click.echo(f"\n📁 Full results location: {stage1_path}")


def start_stage2_execution(project: BrandAuditProject) -> None:
   """Start Stage 2 manual execution."""
   
   exec_guide_path = project.project_path / "stage2_execution" / "manual_execution_guide.md"
   
   if not exec_guide_path.exists():
       click.echo("❌ Execution guide not found. Please run Stage 1 first.")
       return
   
   click.echo(f"\n📋 Stage 2: Manual Execution Guide")
   click.echo("=" * 40)
   
   # Update project status
   project.update_stage_status("stage2", StageStatusEnum.IN_PROGRESS)
   
   click.echo("✅ Stage 2 marked as started.")
   click.echo(f"📖 Execution guide: {exec_guide_path}")
   click.echo(f"📁 Save responses to: {project.project_path / 'stage2_execution'}")
   
   # Show key instructions
   try:
       with open(exec_guide_path) as f:
           guide_content = f.read()
       
       # Extract key sections (simplified)
       if "Expected Time" in guide_content:
           lines = guide_content.split('\n')
           for i, line in enumerate(lines):
               if "Expected Time" in line:
                   click.echo(f"\n⏱️  {line.strip()}")
                   break
       
       click.echo(f"\n💡 When finished, run: brandscope upload-results {project.project_id}")
       
   except OSError:
       pass


def view_project_files(project: BrandAuditProject) -> None:
   """View project file structure."""
   
   click.echo(f"\n📁 Project Files - {project.display_name}")
   click.echo("=" * 50)
   
   def show_directory(path: Path, prefix: str = "") -> None:
       try:
           items = list(path.iterdir())
           items.sort(key=lambda x: (x.is_file(), x.name))
           
           for item in items:
               if item.name.startswith('.'):
                   continue
                   
               if item.is_dir():
                   click.echo(f"{prefix}📁 {item.name}/")
                   if len(list(item.iterdir())) > 0:
                       show_directory(item, prefix + "  ")
               else:
                   size_kb = item.stat().st_size / 1024
                   click.echo(f"{prefix}📄 {item.name} ({size_kb:.1f} KB)")
       except OSError:
           pass
   
   show_directory(project.project_path)
   
   click.echo(f"\n📂 Project location: {project.project_path}")


def export_project_results(project: BrandAuditProject) -> None:
   """Export project results."""
   
   click.echo(f"\n📤 Export Results - {project.display_name}")
   click.echo("Export functionality will be implemented in Stage 3.")


def show_project_settings(project: BrandAuditProject) -> None:
   """Show project settings."""
   
   config = project.config
   
   click.echo(f"\n⚙️  Project Settings - {project.display_name}")
   click.echo("=" * 50)
   
   click.echo(f"Project ID: {config.project_metadata.project_id}")
   click.echo(f"Brand: {config.project_metadata.brand}")
   click.echo(f"Category: {config.project_metadata.category}")
   click.echo(f"Created: {config.project_metadata.created_date}")
   click.echo(f"Last Modified: {config.project_metadata.last_modified}")
   click.echo(f"Version: {config.project_metadata.version}")
   
   # Show stage status
   click.echo(f"\nStage Status:")
   for stage, status in config.stage_status.items():
       click.echo(f"  {stage}: {status.status.value}")
       if status.error_message:
           click.echo(f"    Error: {status.error_message}")


if __name__ == '__main__':
   cli()