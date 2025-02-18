# Import main classes from each module
from .panning import StereoPanning
from .enhancer import StereoEnhancer
from .imager import StereoImager
from .widener import StereoWidener

# Make them available when importing from filters
__all__ = [
    'StereoPanning',
    'StereoEnhancer',
    'StereoImager',
    'StereoWidener'
]