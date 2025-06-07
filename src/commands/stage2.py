# src/commands/stage2.py
"""
Stage 2 command implementation for prompt generation.
CORRECTED: Fixed integration with new prompt injection system.
"""
import asyncio
import click
from ..core.project_manager import BrandAuditProject
from ..stage2.prompt_executor import execute_stage2
from ..utils.logging import get_logger
from ..utils.exceptions import StageExecutionError


logger = get_logger(__name__)


async def execute_stage2_command(project: BrandAuditProject = None) -> None:
    """Execute Stage 2 prompt generation command with updated injection system."""
    
    if not project:
        click.echo("❌ No project context available")
        return
    
    try:
        click.echo(f"\n🚀 Executing Stage 2 for {project.display_name}")
        click.echo("=" * 60)
        
        # Check prerequisites
        if not validate_stage2_prerequisites(project):
            click.echo("❌ Stage 1 must be completed before running Stage 2")
            click.echo("💡 Run Stage 1 first to generate customer archetypes and queries")
            return
        
        click.echo("✅ Stage 1 complete - proceeding with prompt generation")
        click.echo("🔄 Loading Stage 1 data and generating injectable prompts...")
        
        # Execute Stage 2 with the updated prompt injection system
        result = await execute_stage2(project)
        
        # Display success results
        click.echo(f"\n🎉 Stage 2 completed successfully!")
        click.echo(f"📊 Execution Summary:")
        
        # Extract data from the new execution package format
        execution_package = result['execution_package']
        metadata = execution_package['metadata']
        execution_summary = execution_package['execution_summary']
        
        click.echo(f"   • Prompts generated: {metadata['prompts_generated']}")
        click.echo(f"   • Priority queries: {execution_summary['total_queries']}")
        click.echo(f"   • Target platforms: {len(execution_package.get('platforms', []))}")
        click.echo(f"   • Expected response files: {execution_summary['expected_response_files']}")
        click.echo(f"   • Estimated execution time: {execution_summary['estimated_execution_time_minutes']} minutes")
        
        # Show customer context
        if execution_package.get('customer_context'):
            context_preview = execution_package['customer_context'][:100]
            click.echo(f"\n👤 Customer Context:")
            click.echo(f"   {context_preview}...")
        
        # Show execution matrix preview
        if execution_package.get('execution_matrix'):
            click.echo(f"\n🎯 Query Execution Preview:")
            for i, matrix_entry in enumerate(execution_package['execution_matrix'][:3], 1):
                click.echo(f"   {i}. {matrix_entry['styled_query'][:50]}...")
                click.echo(f"      Archetype: {matrix_entry['archetype']}")
                click.echo(f"      Priority: {matrix_entry['execution_priority']}")
        
        # Show file artifacts
        click.echo(f"\n📂 Generated Files:")
        for artifact_name, file_path in result['artifacts'].items():
            # Get file size
            try:
                from pathlib import Path
                size_kb = Path(file_path).stat().st_size / 1024
                click.echo(f"   • {artifact_name}: {Path(file_path).name} ({size_kb:.1f} KB)")
            except:
                click.echo(f"   • {artifact_name}: {file_path}")
        
        # Show directory structure created
        click.echo(f"\n📁 Directory Structure Created:")
        stage2_path = project.get_file_path("stage2_execution")
        click.echo(f"   stage2_execution/")
        click.echo(f"   ├── execution_package_*.json       (Machine-readable prompts)")
        click.echo(f"   ├── manual_execution_guide_*.md    (Human-readable guide)")
        click.echo(f"   ├── prompts/")
        click.echo(f"   │   ├── natural_prompts_*.json     (Natural AI prompts)")
        click.echo(f"   │   └── controlled_prompts_*.json  (Controlled AI prompts)")
        click.echo(f"   ├── natural_dataset/               (Save natural responses here)")
        click.echo(f"   └── controlled_dataset/            (Save controlled responses here)")
        
        # Show next steps with specific file references
        click.echo(f"\n📋 Next Steps:")
        click.echo(f"1. 📖 Read the execution guide:")
        guide_file = result['artifacts'].get('execution_guide', 'manual_execution_guide_*.md')
        click.echo(f"   • Open: {guide_file}")
        
        click.echo(f"2. 🤖 Execute prompts manually across AI platforms:")
        click.echo(f"   • Use prompts from: execution_package_*.json")
        click.echo(f"   • Test on: Claude, ChatGPT, Gemini, Grok")
        
        click.echo(f"3. 💾 Save responses using exact naming convention:")
        click.echo(f"   • Natural: natural_dataset/{{platform}}_query{{NN}}_natural.json")
        click.echo(f"   • Controlled: controlled_dataset/{{platform}}_query{{NN}}_controlled.json")
        
        click.echo(f"4. ✅ Quality check:")
        click.echo(f"   • Expected total files: {execution_summary['expected_response_files']}")
        click.echo(f"   • Validate with: brandscope validate-responses {project.project_id}")
        
        click.echo(f"5. 📤 Proceed to Stage 3 analysis when complete")
        
    except StageExecutionError as e:
        logger.error("Stage 2 execution failed", exc_info=True)
        click.echo(f"\n❌ Stage 2 failed: {e.message}")
        if hasattr(e, 'details') and e.details:
            click.echo(f"💡 Details: {e.details}")
        
        # Show recovery options
        click.echo(f"\n🔧 Recovery Options:")
        click.echo(f"• Check Stage 1 outputs are complete:")
        click.echo(f"  - Verify: projects/{project.project_id}/stage1_outputs/")
        click.echo(f"• Verify project configuration integrity")
        click.echo(f"• Try regenerating Stage 1 if files are missing")
        click.echo(f"• Check logs for detailed error information")
        
    except Exception as e:
        logger.error("Unexpected Stage 2 error", exc_info=True)
        click.echo(f"\n💥 Unexpected error: {str(e)}")
        click.echo(f"💡 Check logs for detailed error information")
        
        # Debug information
        click.echo(f"\n🔍 Debug Information:")
        click.echo(f"• Project ID: {project.project_id}")
        click.echo(f"• Project path: {project.project_path}")
        stage1_path = project.get_file_path("stage1_outputs")
        click.echo(f"• Stage 1 outputs exist: {stage1_path.exists()}")


def validate_stage2_prerequisites(project: BrandAuditProject) -> bool:
    """Validate that Stage 2 can be executed."""
    try:
        # Check Stage 1 completion
        status = project.get_status()
        if not status['stage1_complete']:
            return False
        
        # Check for Stage 1 outputs directory
        stage1_dir = project.get_file_path("stage1_outputs")
        if not stage1_dir.exists():
            return False
        
        # Check for execution package or styled queries (new system is flexible)
        execution_packages = list(stage1_dir.glob("*execution_package*.json"))
        styled_queries = list(stage1_dir.glob("*styled_queries*.json"))
        customer_archetypes = list(stage1_dir.glob("*customer_archetypes*.json"))
        
        # Need at least queries and archetypes for the new system
        return len(styled_queries) > 0 and len(customer_archetypes) > 0
        
    except Exception as e:
        logger.error(f"Stage 2 prerequisite validation failed: {e}")
        return False


def show_stage2_status(project: BrandAuditProject) -> None:
    """Show detailed Stage 2 status information with new file structure."""
    stage2_path = project.get_file_path("stage2_execution")
    
    click.echo(f"\n📊 Stage 2 Status for {project.display_name}")
    click.echo("=" * 50)
    
    if not stage2_path.exists():
        click.echo("📁 Stage 2 directory: Not created")
        
        # Check if prerequisites are met
        if validate_stage2_prerequisites(project):
            click.echo("🚀 Status: Ready to generate prompts")
        else:
            click.echo("⚠️  Status: Stage 1 must be completed first")
        return
    
    # Check for generated files (new format)
    execution_packages = list(stage2_path.glob("execution_package_*.json"))
    execution_guides = list(stage2_path.glob("manual_execution_guide_*.md"))
    prompt_files = list((stage2_path / "prompts").glob("*.json")) if (stage2_path / "prompts").exists() else []
    
    click.echo(f"📁 Stage 2 directory: Created")
    click.echo(f"📦 Execution packages: {len(execution_packages)}")
    click.echo(f"📖 Execution guides: {len(execution_guides)}")
    click.echo(f"📄 Prompt files: {len(prompt_files)}")
    
    # Check manual execution status
    natural_dir = stage2_path / "natural_dataset"
    controlled_dir = stage2_path / "controlled_dataset"
    
    natural_responses = list(natural_dir.glob("*.json")) if natural_dir.exists() else []
    controlled_responses = list(controlled_dir.glob("*.json")) if controlled_dir.exists() else []
    
    click.echo(f"\n🤖 Manual Execution Status:")
    click.echo(f"   • Natural responses: {len(natural_responses)}")
    click.echo(f"   • Controlled responses: {len(controlled_responses)}")
    
    total_responses = len(natural_responses) + len(controlled_responses)
    
    # Calculate expected responses from execution package
    if execution_packages:
        try:
            import json
            with open(execution_packages[0]) as f:
                package = json.load(f)
            expected = package.get('execution_summary', {}).get('expected_response_files', 0)
            
            if total_responses >= expected:
                click.echo("   ✅ Status: Manual execution complete")
                click.echo(f"   🚀 Ready for Stage 3 analysis")
            elif total_responses > 0:
                completion = (total_responses / expected * 100) if expected > 0 else 0
                click.echo(f"   ⏳ Status: Manual execution {completion:.1f}% complete ({total_responses}/{expected})")
                remaining = expected - total_responses
                click.echo(f"   📋 Remaining files needed: {remaining}")
            else:
                click.echo("   📋 Status: Ready for manual execution")
                click.echo(f"   📖 Follow guide: {execution_guides[0].name if execution_guides else 'manual_execution_guide_*.md'}")
        except Exception as e:
            if total_responses > 0:
                click.echo(f"   ⏳ Status: Manual execution in progress ({total_responses} files)")
            else:
                click.echo("   📋 Status: Ready for manual execution")
    else:
        if total_responses > 0:
            click.echo(f"   ⏳ Status: Manual execution in progress ({total_responses} files)")
        else:
            click.echo("   ⚠️  Status: No execution package found - regenerate Stage 2")
    
    # Show file structure if responses exist
    if total_responses > 0:
        click.echo(f"\n📁 Response Files Structure:")
        
        # Show natural responses
        if natural_responses:
            click.echo(f"   natural_dataset/ ({len(natural_responses)} files)")
            for response_file in sorted(natural_responses)[:3]:
                click.echo(f"   ├── {response_file.name}")
            if len(natural_responses) > 3:
                click.echo(f"   └── ... ({len(natural_responses) - 3} more)")
        
        # Show controlled responses  
        if controlled_responses:
            click.echo(f"   controlled_dataset/ ({len(controlled_responses)} files)")
            for response_file in sorted(controlled_responses)[:3]:
                click.echo(f"   ├── {response_file.name}")
            if len(controlled_responses) > 3:
                click.echo(f"   └── ... ({len(controlled_responses) - 3} more)")