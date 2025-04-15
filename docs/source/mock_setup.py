import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Create numpy mock with version
class MockNumpy(Mock):
    __version__ = '1.24.0'

# Mock numba to avoid the initialization error
class MockNumba(Mock):
    config = Mock()
    # Mock the problematic function
    def _ensure_critical_deps(*args, **kwargs):
        pass

# List all modules to mock
MOCK_MODULES = [
    'numpy', 
    'numba', 'numba.config', 'numba.cuda',
    'torch', 'torch.nn', 'torch.utils',
    'pynvjitlink-cu12'
]

# Apply generic mocks
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Apply specific mocks
sys.modules['numpy'] = MockNumpy()
sys.modules['numba'] = MockNumba()