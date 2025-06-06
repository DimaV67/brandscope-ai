"""
Stage 1 command implementation.
"""
import asyncio
import click
from ..core.project_manager import BrandAuditProject
from ..stage1.prompt_generator import Stage1Generator
from ..utils.logging import get_logger
from ..utils.exceptions import StageExecutionError


logger = get_logger(__name__)


def execute_stage1(project: BrandAuditProject = None) -> None:
    """Execute Stage 1 prompt generation."""
    
    if not project:
        click.echo("❌ No project context available")
        return
    
    try:
        click.echo(f"\n🔄 Executing Stage 1 for {project.display_name}")
        click.echo("=" * 50)
        
        # Create generator
        generator = Stage1Generator(project)
        
        # The logger will now print live updates from the pipeline.
        click.echo("🚀 Generating prompts... (This may take several minutes)")
        result = asyncio.run(generator.execute_full_pipeline())
        
        # Show results
        click.echo(f"\n✅ Stage 1 completed successfully!")
        click.echo(f"📊 Generated {len(result['execution_package']['customer_archetypes'])} customer archetypes")
        click.echo(f"❓ Created {len(result['execution_package']['execution_queries'])} styled queries")
        click.echo(f"📁 Saved {len(result['artifacts'])} artifact files")
        
        # Show next steps
        click.echo(f"\n📋 Next Steps:")
        click.echo(f"1. Review execution guide: stage2_execution/manual_execution_guide.md")
        click.echo(f"2. Execute manual testing across AI platforms")
        click.echo(f"3. Upload results when complete")
        
        # Show file locations
        click.echo(f"\n📂 Files created:")
        for artifact_name, file_path in result['artifacts'].items():
            click.echo(f"   {artifact_name}: {file_path}")
        
    except StageExecutionError as e:
        logger.error("Stage 1 execution failed", exc_info=True)
        click.echo(f"❌ Stage 1 failed: {e.message}")
        if e.details:
            click.echo(f"Details: {e.details}")
    except Exception as e:
        logger.error("Unexpected Stage 1 error", exc_info=True)
        click.echo(f"💥 Unexpected error: {str(e)}")