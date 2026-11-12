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


class PerlinNoiseBlending(FractalNoiseBlending):
    """A class that blends fractal perlin noise into
           a grayscale noise image based on the proportion of white.

           Args:
                grid (int): The number of grid.
                t (None or float): Represents the elapsed time. If set to None, a random value is generated.
                noise_intensity (float):
                    Adjusting the elevation difference.
                    The larger the value, the greater the elevation difference.
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
        prefix = '8bit' if arr.dtype == np.uint8 else '16bit'
        output_image.output(arr, f'{prefix}_fractal_perlin_blending', parent, with_suffix)

    def blend(self, source):
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
