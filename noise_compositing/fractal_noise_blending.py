import random

import cv2
import numpy as np

from noise import Fractal2D, PerlinNoise, SimplexNoise
from texture_generator.utils import output_image
from texture_generator.utils.noise_process_mixin import NoiseProcessMixin


class FractalNoiseBlending(NoiseProcessMixin):

    def __init__(self, noise_intensity, gain, lacunarity, octaves, debug):
        self.noise_intensity = noise_intensity
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.debug = debug

    def blend_fractal(self, source_arr, noise_arr):
        source_max_val = np.iinfo(source_arr.dtype).max
        source_dtype = source_arr.dtype

        source_arr = source_arr.astype(np.float32)
        noise_arr = noise_arr.astype(np.float32)
        noise_arr = (noise_arr - noise_arr.min()) / (noise_arr.max() - noise_arr.min())

        if self.debug:
            output_image.output(self.clip_array(noise_arr), 'blended_fractal_noise')

        mask = source_arr / source_max_val
        layered_arr = source_arr - (noise_arr * self.noise_intensity * mask)

        layered_arr = np.clip(layered_arr, 0, source_max_val).astype(source_dtype)
        return layered_arr

    def get_array(self, source):
        match source:
            case str():
                if (img := cv2.imread(source, cv2.IMREAD_UNCHANGED)) is None:
                    raise FileNotFoundError(f'Not found or could not be read: {source}')
                return self.get_array(img)
            case np.ndarray(ndim=2, dtype=d) if d in (np.uint8, np.uint16):
                return source
            case np.ndarray():
                raise ValueError(
                    f'Invalid shape or dtype (ndim={source.ndim}, dtype={source.dtype}). '
                    'The image must be grayscale (2D) and unnormalized (uint8 or uint16).'
                )
            case _:
                raise ValueError(f'Invalid source value: {source}')

    def write_array_to_file(self, arr, file_name, parent, with_suffix):
        prefix = '8bit' if arr.dtype == np.uint8 else '16bit'
        output_image.output(arr, f'{prefix}_{file_name}', parent, with_suffix)


class PerlinNoiseBlending(FractalNoiseBlending):
    """A class that blends fractal perlin noise into a grayscale noise image based on the proportion of white.

        Args:
            grid (int): The number of grid.
            t (None or float): The elapsed time. If set to None, a random value is generated. Defalt is None.
            noise_intensity (float):
                Adjusting the elevation difference. The larger the value, the greater the elevation difference.
                When processing 16-bit images, it's good to specify a fairly large value.
            gain (float):
                A coefficient that determines by how many times the
                amplitude (strength or height of the effect) is multiplied for each octave shift.
            lacunarity (float):
                A multiplier that determines how many times the frequency (scale/resolution)
                is increased with each octave step.
            octaves (int):
                A number indicating how many layers (or times) of noise are superimposed.
            debug (bool): If True, an image of blended noise is output.
    """

    def __init__(self, grid=4, t=None, noise_intensity=100,
                 gain=0.5, lacunarity=2.01, octaves=4, debug=False):
        super().__init__(noise_intensity, gain, lacunarity, octaves, debug)
        self.grid = grid
        self.t = t

    def output_image(self, source, parent='.', with_suffix=True):
        arr = self.blend(source)
        file_name = 'fractal_perlin_blending'
        self.write_array_to_file(arr, file_name, parent, with_suffix)

    def blend(self, source):
        """Blend fractal simplex noise into the original image.
            Args:
                source (str or numpy.ndarray):
                    Either the file path to a grayscale noise image or
                    the noise image converted to an ndarray (before normalization).
        """
        source_arr = self.get_array(source)
        size = max(source_arr.shape)

        perlin = PerlinNoise()
        noise_arr = perlin.fractal2(
            size=size,
            grid=self.grid,
            t=self.t,
            gain=self.gain,
            lacunarity=self.lacunarity,
            octaves=self.octaves,
        )

        return self.blend_fractal(source_arr, noise_arr)


class SimplexNoiseBlending(FractalNoiseBlending):
    """A class that blends fractal simplex noise into a grayscale noise image based on the proportion of white.

        Args:
            noise_scale (float): The higher the value, the gentler the slope becomes.
            t (None or float): The elapsed time. If set to None, a random value is generated. Defalt is None.
            noise_intensity (float):
                Adjusting the elevation difference. The larger the value, the greater the elevation difference.
                When processing 16-bit images, it's good to specify a fairly large value.
            gain (float):
                A coefficient that determines by how many times the
                amplitude (strength or height of the effect) is multiplied for each octave shift.
            lacunarity (float):
                A multiplier that determines how many times the frequency (scale/resolution)
                is increased with each octave step.
            octaves (int):
                A number indicating how many layers (or times) of noise are superimposed.
            debug (bool): If True, an image of blended noise is output.
    """

    def __init__(self, noise_scale=200, t=None, noise_intensity=100,
                 gain=0.5, lacunarity=2.01, octaves=4, debug=False):
        super().__init__(noise_intensity, gain, lacunarity, octaves, debug)
        self.noise_scale = noise_scale
        self.t = t

    def output_image(self, source, parent='.', with_suffix=True):
        arr = self.blend(source)
        file_name = 'fractal_simplex_blending'
        self.write_array_to_file(arr, file_name, parent, with_suffix)

    def blend(self, source):
        """Blend fractal simplex noise into the original image.
            Args:
                source (str or numpy.ndarray):
                    Either the file path to a grayscale noise image or
                    the noise image converted to an ndarray (before normalization).
        """
        source_arr = self.get_array(source)
        size = max(source_arr.shape)

        t = self.t if self.t is not None else random.uniform(0, 1000)
        simplex = SimplexNoise()

        noise = Fractal2D(
            simplex.snoise2,
            gain=self.gain,
            lacunarity=self.lacunarity,
            octaves=self.octaves
        )

        noise_arr = np.array(
            [noise.fractal(x / self.noise_scale + t, y / self.noise_scale + t)
                for y in range(size) for x in range(size)]
        )

        noise_arr = noise_arr.reshape(size, size)
        return self.blend_fractal(source_arr, noise_arr)