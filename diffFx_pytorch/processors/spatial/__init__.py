# Import main classes from each module
from .panning import StereoPanning
from .enhancer import StereoEnhancer
from .widener import StereoWidener, MultiBandStereoWidener

# Make them available when importing from filters
__all__ = [
    'StereoPanning',
    'StereoEnhancer',
    'StereoWidener',
    'MultiBandStereoWidener'
]