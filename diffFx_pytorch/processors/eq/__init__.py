# Import main classes from each module
from .tonestack import Tonestack
from .parametricEQ import ParametricEqualizer
from .graphicEQ import GraphicEqualizer
# Make them available when importing from filters
__all__ = [
    'Tonestack',
    'ParametricEqualizer',
    'GraphicEqualizer',
]