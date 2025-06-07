#!/usr/bin/env python3
"""
Test script for Stage 1 → Stage 2 data pipeline integration.
Run from project root directory.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.stage2.stage1_data_loader import Stage1DataLoader
from src.core.project_manager import ProjectManager
from src.utils.logging import setup_logging

def setup_test_logging():
    """Setup logging for test visibility."""
    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

async def test_stage1_discovery():
    """Test Stage 1 file discovery."""
    print("🔍 Testing Stage 1 file discovery...")
    
    try:
        project_manager = ProjectManager()
        project = project_manager.load_project("wonderful_pistachios_20250605")
        
        loader = Stage1DataLoader(project)
        
        # Test file discovery
        stage1_files = loader._discover_stage1_files()
        
        print(f"✅ Found {len(stage1_files)} Stage 1 file types:")
        for file_type, file_path in stage1_files.items():
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"   - {file_type}: {file_path.name} ({file_size:.1f} KB)")
        
        return stage1_files
        
    except Exception as e:
        print(f"❌ File discovery failed: {e}")
        return None

async def test_stage1_data_loading(stage1_files):
    """Test Stage 1 data loading and validation."""
    print("\n📊 Testing Stage 1 data loading...")
    
    try:
        project_manager = ProjectManager()
        project = project_manager.load_project("wonderful_pistachios_20250605")
        
        loader = Stage1DataLoader(project)
        
        # Load Stage 1 data
        stage1_data = loader.load_stage1_outputs()
        
        print(f"✅ Stage 1 data loaded successfully:")
        print(f"   - Archetypes: {len(stage1_data.customer_archetypes)}")
        print(f"   - Execution queries: {len(stage1_data.execution_queries)}")
        print(f"   - Metadata keys: {list(stage1_data.metadata.keys())}")
        
        # Show archetype details
        if stage1_data.customer_archetypes:
            print(f"\n📋 Customer Archetypes:")
            for i, archetype in enumerate(stage1_data.customer_archetypes[:3], 1):
                print(f"   {i}. {archetype.get('name', 'Unknown')}")
                print(f"      - Strategic Value: {archetype.get('strategic_value', 'Unknown')}")
                print(f"      - Confidence: {archetype.get('confidence', 'Unknown')}")
        
        # Show query details
        if stage1_data.execution_queries:
            print(f"\n❓ Top 5 Execution Queries:")
            for i, query in enumerate(stage1_data.execution_queries[:5], 1):
                print(f"   {i}. {query.get('styled_query', 'No query text')[:60]}...")
                print(f"      - Priority: {query.get('execution_priority', 'Unknown')}")
                print(f"      - Authenticity: {query.get('authenticity_score', 'Unknown')}")
        
        return stage1_data
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_priority_query_selection(stage1_data):
    """Test priority query selection."""
    print("\n🎯 Testing priority query selection...")
    
    try:
        project_manager = ProjectManager()
        project = project_manager.load_project("wonderful_pistachios_20250605")
        
        loader = Stage1DataLoader(project)
        
        # Test priority query selection
        priority_queries = loader.get_priority_queries(stage1_data, max_queries=5)
        
        print(f"✅ Selected {len(priority_queries)} priority queries:")
        for i, query in enumerate(priority_queries, 1):
            print(f"   {i}. {query.get('styled_query', 'No query text')[:50]}...")
            print(f"      - Authenticity: {query.get('authenticity_score', 'Unknown')}")
            print(f"      - Archetype: {query.get('archetype', 'Unknown')}")
        
        return priority_queries
        
    except Exception as e:
        print(f"❌ Priority query selection failed: {e}")
        return None

async def test_customer_context_extraction(stage1_data):
    """Test customer context extraction."""
    print("\n👤 Testing customer context extraction...")
    
    try:
        project_manager = ProjectManager()
        project = project_manager.load_project("wonderful_pistachios_20250605")
        
        loader = Stage1DataLoader(project)
        
        # Test customer context extraction
        customer_context = loader.extract_customer_context(stage1_data)
        
        print(f"✅ Customer context extracted:")
        print(f"   Length: {len(customer_context)} characters")
        print(f"   Preview: {customer_context[:200]}...")
        
        return customer_context
        
    except Exception as e:
        print(f"❌ Customer context extraction failed: {e}")
        return None

async def test_validation():
    """Test Stage 2 validation."""
    print("\n✅ Testing Stage 2 validation...")
    
    try:
        from src.stage2.validation import Stage2Validator
        
        project_manager = ProjectManager()
        project = project_manager.load_project("wonderful_pistachios_20250605")
        
        loader = Stage1DataLoader(project)
        stage1_data = loader.load_stage1_outputs()
        
        # Test validation
        validator = Stage2Validator()
        is_valid = validator.validate_stage1_inputs(stage1_data.__dict__)
        
        if is_valid:
            print("✅ Stage 1 data validation passed")
        else:
            print("❌ Stage 1 data validation failed:")
            for error in validator.validation_errors:
                print(f"   - Error: {error}")
            for warning in validator.validation_warnings:
                print(f"   - Warning: {warning}")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Stage 1 → Stage 2 Integration Test")
    print("=" * 50)
    
    setup_test_logging()
    
    # Test 1: File discovery
    stage1_files = await test_stage1_discovery()
    if not stage1_files:
        print("\n❌ Cannot proceed - Stage 1 files not found")
        return False
    
    # Test 2: Data loading
    stage1_data = await test_stage1_data_loading(stage1_files)
    if not stage1_data:
        print("\n❌ Cannot proceed - Stage 1 data loading failed")
        return False
    
    # Test 3: Priority query selection
    priority_queries = await test_priority_query_selection(stage1_data)
    if not priority_queries:
        print("\n❌ Priority query selection failed")
        return False
    
    # Test 4: Customer context extraction
    customer_context = await test_customer_context_extraction(stage1_data)
    if not customer_context:
        print("\n❌ Customer context extraction failed")
        return False
    
    # Test 5: Validation
    validation_passed = await test_validation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Integration Test Summary:")
    print(f"   ✅ File Discovery: Success")
    print(f"   ✅ Data Loading: Success")
    print(f"   ✅ Query Selection: Success ({len(priority_queries)} queries)")
    print(f"   ✅ Context Extraction: Success ({len(customer_context)} chars)")
    print(f"   {'✅' if validation_passed else '❌'} Validation: {'Passed' if validation_passed else 'Failed'}")
    
    if validation_passed:
        print("\n🎉 All tests passed! Stage 1 → Stage 2 pipeline is ready.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)