# Import main classes from each module
from .enhancer import Enhancer
from .imager import Imager
from .widener import Widener

# Make them available when importing from filters
__all__ = [
    'Enhancer',
    'Imager',
    'Widener'
]