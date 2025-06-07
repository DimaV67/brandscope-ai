# src/commands/stage2.py
"""
Stage 2 command implementation for prompt generation.
"""
import asyncio
import click
from ..core.project_manager import BrandAuditProject
from ..stage2.prompt_executor import execute_stage2
from ..utils.logging import get_logger
from ..utils.exceptions import StageExecutionError


logger = get_logger(__name__)


def execute_stage2_command(project: BrandAuditProject = None) -> None:
    """Execute Stage 2 prompt generation command."""
    
    if not project:
        click.echo("❌ No project context available")
        return
    
    try:
        click.echo(f"\n🚀 Executing Stage 2 for {project.display_name}")
        click.echo("=" * 60)
        
        # Check prerequisites
        status = project.get_status()
        if not status['stage1_complete']:
            click.echo("❌ Stage 1 must be completed before running Stage 2")
            click.echo("💡 Run Stage 1 first to generate customer archetypes and queries")
            return
        
        click.echo("✅ Stage 1 complete - proceeding with prompt generation")
        click.echo("🔄 Generating natural and controlled AI prompts...")
        
        # Execute Stage 2
        result = asyncio.run(execute_stage2(project))
        
        # Display success results
        click.echo(f"\n🎉 Stage 2 completed successfully!")
        click.echo(f"📊 Generated prompts for {result['priority_queries_count']} priority queries")
        click.echo(f"📁 Created {len(result['artifacts'])} artifact files")
        
        # Show execution details
        click.echo(f"\n📋 Execution Package Details:")
        click.echo(f"   • Natural AI prompts: {result['priority_queries_count']}")
        click.echo(f"   • Controlled AI prompts: {result['priority_queries_count']}")
        click.echo(f"   • Target platforms: 4 (Claude, ChatGPT, Gemini, Grok)")
        click.echo(f"   • Expected response files: {result['priority_queries_count'] * 4 * 2}")
        click.echo(f"   • Estimated execution time: {result['priority_queries_count'] * 4 * 5} minutes")
        
        # Show file artifacts
        click.echo(f"\n📂 Generated Files:")
        for artifact_name, file_path in result['artifacts'].items():
            click.echo(f"   • {artifact_name}: {file_path}")
        
        # Show next steps
        click.echo(f"\n📋 Next Steps:")
        click.echo(f"1. 📖 Review execution guide in stage2_execution/")
        click.echo(f"2. 🤖 Execute prompts manually across AI platforms")
        click.echo(f"3. 💾 Save responses using naming convention:")
        click.echo(f"   • Natural: platform_queryNN_natural.json")
        click.echo(f"   • Controlled: platform_queryNN_controlled.json")
        click.echo(f"4. 📤 Upload results when manual execution is complete")
        
    except StageExecutionError as e:
        logger.error("Stage 2 execution failed", exc_info=True)
        click.echo(f"\n❌ Stage 2 failed: {e.message}")
        if e.details:
            click.echo(f"💡 Details: {e.details}")
            
        # Show recovery options
        click.echo(f"\n🔧 Recovery Options:")
        click.echo(f"• Check Stage 1 outputs are complete")
        click.echo(f"• Verify project configuration")
        click.echo(f"• Try regenerating Stage 2")
        
    except Exception as e:
        logger.error("Unexpected Stage 2 error", exc_info=True)
        click.echo(f"\n💥 Unexpected error: {str(e)}")
        click.echo(f"💡 Check logs for detailed error information")


def validate_stage2_prerequisites(project: BrandAuditProject) -> bool:
    """Validate that Stage 2 can be executed."""
    try:
        # Check Stage 1 completion
        status = project.get_status()
        if not status['stage1_complete']:
            return False
        
        # Check for Stage 1 outputs
        stage1_dir = project.get_file_path("stage1_outputs")
        if not stage1_dir.exists():
            return False
        
        # Check for execution package
        execution_packages = list(stage1_dir.glob("*execution_package*.json"))
        return len(execution_packages) > 0
        
    except Exception as e:
        logger.error(f"Stage 2 prerequisite validation failed: {e}")
        return False


def show_stage2_status(project: BrandAuditProject) -> None:
    """Show detailed Stage 2 status information."""
    stage2_path = project.get_file_path("stage2_execution")
    
    click.echo(f"\n📊 Stage 2 Status for {project.display_name}")
    click.echo("=" * 50)
    
    if not stage2_path.exists():
        click.echo("📁 Stage 2 directory: Not created")
        click.echo("🚀 Status: Ready to generate prompts")
        return
    
    # Check for generated files
    prompt_files = list(stage2_path.glob("*prompts*.json"))
    execution_files = list(stage2_path.glob("*execution*.json"))
    guide_files = list(stage2_path.glob("*guide*.md"))
    
    click.echo(f"📁 Stage 2 directory: Created")
    click.echo(f"📄 Prompt files: {len(prompt_files)}")
    click.echo(f"📋 Execution files: {len(execution_files)}")
    click.echo(f"📖 Guide files: {len(guide_files)}")
    
    # Check manual execution status
    natural_dir = stage2_path / "natural_dataset"
    controlled_dir = stage2_path / "controlled_dataset"
    
    natural_responses = list(natural_dir.glob("*.json")) if natural_dir.exists() else []
    controlled_responses = list(controlled_dir.glob("*.json")) if controlled_dir.exists() else []
    
    click.echo(f"\n🤖 Manual Execution Status:")
    click.echo(f"   • Natural responses: {len(natural_responses)}")
    click.echo(f"   • Controlled responses: {len(controlled_responses)}")
    
    total_responses = len(natural_responses) + len(controlled_responses)
    
    if total_responses == 0:
        click.echo("   📋 Status: Prompts generated, manual execution pending")
    elif execution_files:
        try:
            import json
            with open(execution_files[0]) as f:
                package = json.load(f)
            expected = package.get('execution_summary', {}).get('expected_responses', 0)
            
            if total_responses >= expected:
                click.echo("   ✅ Status: Manual execution complete")
            else:
                completion = (total_responses / expected * 100) if expected > 0 else 0
                click.echo(f"   ⏳ Status: Manual execution {completion:.1f}% complete ({total_responses}/{expected})")
        except:
            click.echo(f"   ⏳ Status: Manual execution in progress ({total_responses} files)")
    else:
        click.echo(f"   ⏳ Status: Manual execution in progress ({total_responses} files)")