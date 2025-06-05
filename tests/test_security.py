"""
Test security utilities.
"""
import pytest
from pathlib import Path

from src.utils.security import SecurityValidator, SecurityConfig


class TestSecurityValidator:
    """Test SecurityValidator."""
    
    @pytest.fixture
    def security_config(self):
        """Create test security config."""
        return SecurityConfig(
            secret_key="test_secret_key_32_chars_long_123",
            api_rate_limit=10,
            max_file_size=1024
        )
    
    @pytest.fixture
    def validator(self, security_config):
        """Create security validator."""
        return SecurityValidator(security_config)
    
    def test_validate_file_path_safe(self, validator, temp_dir):
        """Test safe file path validation."""
        safe_path = Path("subdir/file.txt")
        assert validator.validate_file_path(safe_path, temp_dir) is True
    
    def test_validate_file_path_traversal(self, validator, temp_dir):
        """Test path traversal prevention."""
        dangerous_path = Path("../../../etc/passwd")
        assert validator.validate_file_path(dangerous_path, temp_dir) is False
    
    def test_sanitize_input(self, validator):
        """Test input sanitization."""
        dangerous_input = "<script>alert('xss')</script>test"
        sanitized = validator.sanitize_input(dangerous_input)
        
        # Should remove all dangerous characters
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "'" not in sanitized
        assert '"' not in sanitized
        assert "&" not in sanitized
        
        # Should result in clean text
        expected = "scriptalert(xss)/scripttest"
        assert sanitized == expected
    
    def test_generate_secure_filename(self, validator):
        """Test secure filename generation."""
        dangerous_name = "../../../evil<script>.txt"
        safe_name = validator.generate_secure_filename(dangerous_name)
        
        # Should remove path traversal and dangerous characters
        assert ".." not in safe_name
        assert "/" not in safe_name
        assert "<" not in safe_name
        assert ">" not in safe_name
        
        # Should result in clean filename
        expected = "evil.txt"
        assert safe_name == expected
    
    def test_generate_secure_filename_edge_cases(self, validator):
        """Test secure filename generation edge cases."""
        # Test empty input
        result1 = validator.generate_secure_filename("")
        assert result1 == "safe_file"
        
        # Test only dangerous characters
        result2 = validator.generate_secure_filename("<<<>>>")
        assert result2 == "safe_file"
        
        # Test leading dots
        result3 = validator.generate_secure_filename("...hidden.txt")
        assert result3 == "hidden.txt"
        
        # Test multiple dots - should collapse to single dot
        result4 = validator.generate_secure_filename("file....txt")
        expected4 = "file.txt"  # Multiple dots should become single dot
        assert result4 == expected4, f"Expected '{expected4}', got '{result4}'"
    
    def test_hmac_signature(self, validator):
        """Test HMAC signature creation and verification."""
        data = "test_data"
        signature = validator.create_hmac_signature(data)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
        assert validator.verify_hmac_signature(data, signature) is True
        assert validator.verify_hmac_signature("wrong_data", signature) is False
    
    def test_file_extension_validation(self, validator):
        """Test file extension validation."""
        # Allowed extensions
        assert validator.validate_file_extension(Path("test.json")) is True
        assert validator.validate_file_extension(Path("test.txt")) is True
        assert validator.validate_file_extension(Path("test.md")) is True
        
        # Disallowed extensions
        assert validator.validate_file_extension(Path("test.exe")) is False
        assert validator.validate_file_extension(Path("test.sh")) is False
        assert validator.validate_file_extension(Path("test")) is False  # No extension


if __name__ == "__main__":
    pytest.main([__file__])