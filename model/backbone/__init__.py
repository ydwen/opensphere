from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .sfnet import sfnet4, sfnet10, sfnet20, sfnet36, sfnet64
from .sfnet_deprecated import sfnet4_deprecated, sfnet10_deprecated
from .sfnet_deprecated import sfnet20_deprecated, sfnet36_deprecated
from .sfnet_deprecated import sfnet64_deprecated


__all__ = [
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100',
    'sfnet4', 'sfnet10', 'sfnet20', 'sfnet36', 'sfnet64',
    'sfnet4_deprecated', 'sfnet10_deprecated',
    'sfnet20_deprecated', 'sfnet36_deprecated',
    'sfnet64_deprecated',
]
