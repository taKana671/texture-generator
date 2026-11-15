import random
from abc import ABC, abstractmethod

import cv2
import numpy as np

from noise import Fractal2D, PerlinNoise, SimplexNoise
from texture_generator.utils import output_image
from texture_generator.utils.noise_process_mixin import NoiseProcessMixin


class FractalNoiseBlending(NoiseProcessMixin, ABC):

    def __init__(self, noise_scale, t, noise_intensity, gain, lacunarity, octaves, debug):
        self.noise_scale = noise_scale
        self.t = t
        self.noise_intensity = noise_intensity
        self.gain = gain
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.debug = debug

    @abstractmethod
    def blend(self, source):
        """Takes either a file path or a Numpy.ndarray of grayscale noise images as an argument,
           blends additional noise into them, and returns a Numpy.ndarray.
        """
        pass

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

    def get_source_array(self, source):
        match source:
            case str():
                if (img := cv2.imread(source, cv2.IMREAD_UNCHANGED)) is None:
                    raise FileNotFoundError(f'Not found or could not be read: {source}')
                return self.get_source_array(img)
            case np.ndarray(ndim=2, dtype=d) if d in (np.uint8, np.uint16):
                return source
            case np.ndarray():
                raise ValueError(
                    f'Invalid shape or dtype (ndim={source.ndim}, dtype={source.dtype}). '
                    'The image must be grayscale (2D) and unnormalized (uint8 or uint16).'
                )
            case _:
                raise ValueError(f'Invalid source value: {source}')

    def get_noise_array(self, noise_gen, source_arr):
        h, w = source_arr.shape
        t = self.t if self.t is not None else random.uniform(0, 1000)

        noise = Fractal2D(
            noise_gen,
            gain=self.gain,
            lacunarity=self.lacunarity,
            octaves=self.octaves
        )

        noise_arr = np.array(
            [noise.fractal(x / self.noise_scale + t, y / self.noise_scale + t)
                for y in range(h) for x in range(w)]
        )

        noise_arr = noise_arr.reshape(h, w)
        return noise_arr

    def output_image(self, source, parent='.', with_suffix=True, extension='png'):
        arr = self.blend(source)
        prefix = '8bit' if arr.dtype == np.uint8 else '16bit'
        name = self.__class__.__name__
        output_image.output(arr, f'{prefix}_{name}', parent, with_suffix, extension)


class PerlinNoiseBlending(FractalNoiseBlending):
    """A class that blends fractal perlin noise into a grayscale noise image based on the proportion of white.

        Args:
            noise_scale (float): The higher the value, the gentler the slope becomes. Default is 200.
            t (None or float): The elapsed time. If set to None, a random value is generated. Defalt is None.
            noise_intensity (float):
                Adjusting the elevation difference. The larger the value, the greater the elevation difference.
                When processing 16-bit images, it's good to specify a fairly large value. Default is 100.
            gain (float):
                A coefficient that determines by how many times the amplitude (strength or height of the effect)
                is multiplied for each octave shift. Default is 0.5.
            lacunarity (float):
                A multiplier that determines how many times the frequency (scale/resolution) is increased
                with each octave step. Default is 2.01.
            octaves (int):
                A number indicating how many layers (or times) of noise are superimposed. Default is 4.
            debug (bool): If True, an image of blended noise is output. Default is False.
    """

    def __init__(self, noise_scale=200, t=None, noise_intensity=100,
                 gain=0.5, lacunarity=2.01, octaves=4, debug=False):
        super().__init__(noise_scale, t, noise_intensity, gain, lacunarity, octaves, debug)

    def blend(self, source):
        """Blend fractal simplex noise into the original image.
            Args:
                source (str or Numpy.ndarray): Either the file path or Numpy.ndarray of a grayscale noise image.
            Returns:
                Numpy.ndarray: The bit depth and size will be the same as those of the original image, “source.”
        """
        source_arr = self.get_source_array(source)
        perlin = PerlinNoise()
        noise_gen = perlin.pnoise2
        noise_arr = self.get_noise_array(noise_gen, source_arr)
        return self.blend_fractal(source_arr, noise_arr)


class SimplexNoiseBlending(FractalNoiseBlending):
    """A class that blends fractal simplex noise into a grayscale noise image based on the proportion of white.

        Args:
            noise_scale (float): The higher the value, the gentler the slope becomes. Default is 200.
            t (None or float): The elapsed time. If set to None, a random value is generated. Defalt is None.
            noise_intensity (float):
                Adjusting the elevation difference. The larger the value, the greater the elevation difference.
                When processing 16-bit images, it's good to specify a fairly large value. Default is 100.
            gain (float):
                A coefficient that determines by how many times the amplitude (strength or height of the effect)
                is multiplied for each octave shift. Default is 0.5.
            lacunarity (float):
                A multiplier that determines how many times the frequency (scale/resolution) is increased
                with each octave step. Default is 2.01.
            octaves (int):
                A number indicating how many layers (or times) of noise are superimposed. Default is 4.
            debug (bool): If True, an image of blended noise is output. Default is False.
    """

    def __init__(self, noise_scale=200, t=None, noise_intensity=100,
                 gain=0.5, lacunarity=2.01, octaves=4, debug=False):
        super().__init__(noise_scale, t, noise_intensity, gain, lacunarity, octaves, debug)

    def blend(self, source):
        """Blend fractal simplex noise into the original image.
            Args:
                source (str or Numpy.ndarray): Either the file path or Numpy.ndarray of a grayscale noise image.
            Returns:
                Numpy.ndarray: The bit depth and size will be the same as those of the original image, “source.”
        """
        source_arr = self.get_source_array(source)
        simplex = SimplexNoise()
        noise_gen = simplex.snoise2
        noise_arr = self.get_noise_array(noise_gen, source_arr)
        return self.blend_fractal(source_arr, noise_arr)