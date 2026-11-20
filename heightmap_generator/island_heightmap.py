import functools
import random

import numpy as np

from noise import PerlinNoise, SimplexNoise, Fractal2D
from ..mask.radial_gradient_generator import RadialGradientMask
from ..utils import output_image
from ..utils.noise_process_mixin import NoiseProcessMixin


class IslandHeightmap(NoiseProcessMixin):
    """A class to create an island heightmap from fractal noise.
        Args:
            noise (function): Function or method that generates noise.
            size (int): The size of an image. Default is 256.
            t (None or float):
                The elapsed time. If set to None, a random value is generated. Default is None.
                If specify a float value, the same hegihtmap will be generated each time the program runs.
            gain (float):
                A coefficient that determines by how many times the amplitude (strength or height of the effect)
                is multiplied for each octave shift. Default is 0.5.
            lacunarity (float):
                A multiplier that determines how many times the frequency (scale/resolution) is increased
                with each octave step. Default is 2.01.
            octaves (int):
                A number indicating how many layers (or times) of noise are superimposed. Default is 4.
            island_center_h (int):
                y-axis center of an island; must be positive; size // 2, if not specified.
                Default is None.
            island_center_w (int):
                x-axis center of an island; must be positive; size // 2, if not specified.
                Default is None.
            island_size (float): The larger the island_size, the smaller the island become. Default is 2.
            reverse (bool): If False, basin-shaped heightmap is generated. Default is False.
    """

    def __init__(self, noise_gen, size=256, t=None, gain=0.5, lacunarity=2.01, octaves=4,
                 island_center_h=None, island_center_w=None, island_size=2, reverse=False):
        self.noise_gen = noise_gen
        self.size = size
        self.t = t
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.island_center_h = island_center_h
        self.island_center_w = island_center_w
        self.island_size = island_size
        self.reverse = reverse

    @classmethod
    def from_sfractal(cls, size=256, t=None, gain=0.5, lacunarity=2.01, octaves=4,
                      island_center_h=None, island_center_w=None, island_size=2, reverse=False):
        simplex = SimplexNoise()
        return cls(simplex.snoise2, size, t, gain, lacunarity, octaves, island_center_h, island_center_w, island_size, reverse)

    @classmethod
    def from_pfractal(cls, size=256, t=None, gain=0.5, lacunarity=2.01, octaves=4,
                      island_center_h=None, island_center_w=None, island_size=2, reverse=False):
        perlin = PerlinNoise()
        return cls(perlin.pnoise2, size, t, gain, lacunarity, octaves, island_center_h, island_center_w, island_size, reverse)

    def output_heightmap(self, parent='.', with_suffix=True, extension='png', bit8=True):
        bit = 8 if bit8 else 16
        heightmap = self.create_heightmap(bit)
        filename = f'{bit}bit_{self.__class__.__name__}_{self.noise_gen.__self__.__class__.__name__}'
        output_image.output(heightmap, filename, parent, with_suffix, extension)

    def create_heightmap(self, bit=None):
        heightmap_arr = self.generate_heightmap()

        if bit is None:
            return heightmap_arr

        heightmap_arr = self.clip_array(heightmap_arr, bit)
        return heightmap_arr

    def get_height(self, mask, x, y, noise_v):
        mask_v = mask.get_gradient(x, y)[0]
        v = 0 if mask_v >= noise_v else noise_v - mask_v
        return v

    def generate_heightmap(self):
        mask = RadialGradientMask(
            height=self.size,
            width=self.size,
            center_h=self.island_center_h,
            center_w=self.island_center_w,
            gradient_size=self.island_size,
            inner_to_outer=not self.reverse
        )

        noise = Fractal2D(self.noise_gen, self.gain, self.lacunarity, self.octaves)
        t = self.t if self.t is not None else random.uniform(0, 1000)
        get_height = functools.partial(self.get_height, mask)

        heightmap_arr = np.array(
            [get_height(x, y, noise.fractal(x / self.size + t, y / self.size + t))
                for y in range(self.size) for x in range(self.size)]
        )
        heightmap_arr = heightmap_arr.reshape(self.size, self.size)
        return heightmap_arr