"""
Test the filename sanitization fix.
"""
import re

def generate_secure_filename(original_name: str) -> str:
    """Generate secure filename."""
    # Remove path traversal patterns
    safe_name = original_name.replace('../', '').replace('..\\', '')
    
    # Remove HTML/script tags completely
    safe_name = re.sub(r'<script[^>]*>.*?</script>', '', safe_name, flags=re.IGNORECASE)
    safe_name = re.sub(r'<[^>]*>', '', safe_name)
    
    # Remove other dangerous characters but keep letters, numbers, dots, hyphens, underscores
    safe_name = re.sub(r'[^a-zA-Z0-9.\-_]', '', safe_name)
    
    # Remove leading dots
    safe_name = safe_name.lstrip('.')
    
    # Handle empty result
    if not safe_name:
        safe_name = 'safe_file'
    
    return safe_name[:100]

def test_filename_cases():
    """Test various filename cases."""
    test_cases = [
        ("../../../evil<script>.txt", "evil.txt"),
        ("<script>alert('xss')</script>file.txt", "file.txt"),
        ("normal_file.txt", "normal_file.txt"),
        ("...hidden.txt", "hidden.txt"),
        ("", "safe_file"),
        ("<<<>>>", "safe_file"),
        ("file....txt", "file....txt"),  # Multiple dots are OK
    ]
    
    for input_name, expected in test_cases:
        result = generate_secure_filename(input_name)
        status = "✅" if result == expected else "❌"
        print(f"{status} Input: '{input_name}' → Output: '{result}' (Expected: '{expected}')")
        
        if result != expected:
            print(f"   MISMATCH: Got '{result}', Expected '{expected}'")

if __name__ == "__main__":
    test_filename_cases()