"""
Debug the filename generation.
"""
import re

def debug_generate_secure_filename(original_name: str) -> str:
    """Debug version with print statements."""
    print(f"Input: '{original_name}'")
    
    # Step 1: Remove path traversal patterns
    safe_name = original_name.replace('../', '').replace('..\\', '')
    print(f"After path removal: '{safe_name}'")
    
    # Step 2: Remove HTML/script tags
    safe_name = re.sub(r'<script[^>]*>.*?</script>', '', safe_name, flags=re.IGNORECASE)
    safe_name = re.sub(r'<[^>]*>', '', safe_name)
    print(f"After HTML removal: '{safe_name}'")
    
    # Step 3: Keep only safe characters
    safe_name = re.sub(r'[^a-zA-Z0-9.\-_]', '', safe_name)
    print(f"After character filtering: '{safe_name}'")
    
    # Step 4: Remove leading dots
    safe_name = safe_name.lstrip('.')
    print(f"After leading dot removal: '{safe_name}'")
    
    # Step 5: Clean up multiple dots
    safe_name = re.sub(r'\.{2,}', '.', safe_name)
    print(f"After dot cleanup: '{safe_name}'")
    
    # Step 6: Handle empty
    if not safe_name:
        safe_name = 'safe_file'
        print(f"After empty check: '{safe_name}'")
    
    result = safe_name[:100]
    print(f"Final result: '{result}'")
    return result

def test_debug():
    """Test problematic cases."""
    test_cases = [
        "file....txt",
        "...hidden.txt", 
        "<<<>>>",
        "",
        "../../../evil<script>.txt"
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        result = debug_generate_secure_filename(case)
        print(f"FINAL: '{case}' → '{result}'")

if __name__ == "__main__":
    test_debug()