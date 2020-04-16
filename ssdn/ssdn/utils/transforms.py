import ssdn
from typing import NewType


Transform = NewType("Transform", object)
"""Typing label for otherwise undefined Transform type.
"""


class NoiseTransform(object):
    def __init__(self, style: str):
        self.style = style

    def __call__(self, imgs):
        imgs, params = ssdn.utils.noise.add_style(imgs, self.style)
        return imgs
