import random

import cv2
import numpy as np

from noise import VoronoiEdges, VoronoiRoundEdges
from texture_generator.utils import output_image
from texture_generator.utils.noise_process_mixin import NoiseProcessMixin


class RockPillarsHeightmap(NoiseProcessMixin):

    """A class for generating terrain heightmap like rock pillars.
        Args:
            cliff_area (float):
                Specify how far from the edge of the Voronoi cell the gradient should extend.
                Default is 0.4.
            random_height (bool):
                If True, the height of the cliff tops will all be set to 1.0.
                Otherwise, they will be selected randomly from RockPillarsHeightmap.rock_heights. Default is False.
            blur (int):
                Specify None or an odd number. Negative values are not allowed. Default is 11.
            exclude_edge_cells (bool):
                If True, exclude Voronoi regions that are cut off at the edges of the image. Default is True
    """

    rock_heights = [0.8, 0.9, 1.0]

    def __init__(self, cliff_area=0.4, random_height=False, blur=11, exclude_edge_cells=True):
        self.cliff_area = cliff_area
        self.random_height = random_height
        self.ksize = self.convert_to_ksize(blur)
        self.exclude_edge_cells = exclude_edge_cells

    def convert_to_ksize(self, v):
        if v is not None:
            if v < 0 or (v > 0 and v % 2 == 0):
                raise ValueError("specify a positive odd number.")
        return v

    def generate_heightmap(self, source_arr):
        """Create heightmap from a voronoi noise image.
            Args:
                source_arr (numpy.ndarray): 8-bit, single-channel (binary) source image.
            Returns:
                numpy.ndarray heightmap image with dtype float32, having the same size with the source_arr.
        """
        h, w = source_arr.shape[:2]
        heightmap = np.zeros((h, w), dtype=np.float32)

        _, thresh = cv2.threshold(source_arr, 127, 255, cv2.THRESH_BINARY)
        # Creating a distance map from each pixel to the nearest background (pixels with a value of 0)
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        # Region detection
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] < 10:
                continue

            y_indices, x_indices = np.where(labels == label_idx)

            if self.exclude_edge_cells:
                coords = np.column_stack([y_indices, x_indices])
                targets = [0, 255]

                if np.any(np.isin(coords, targets)):
                    continue

            cell_dists = dist_transform[y_indices, x_indices]

            # Get the maximum distance within a cell (the deepest center).
            max_dist = np.max(cell_dists)
            if max_dist == 0:
                max_dist = 1.0

            flat_threshold = max_dist * self.cliff_area
            max_val = 1.0

            if self.random_height:
                max_val = random.choice(RockPillarsHeightmap.rock_heights)

            cushion = np.clip(cell_dists / flat_threshold, 0.0, max_val)
            heightmap[y_indices, x_indices] = cushion

        if self.ksize is not None:
            heightmap = cv2.GaussianBlur(heightmap, (self.ksize, self.ksize), 0)

        return heightmap

    def output_heightmap(self, file_path, parent='.', with_suffix=True, extension='png', bit8=True):
        """Create a heightmap from a Voronoi noise image file.
            Arges:
                file_path (str):
                    A file pash of 8-bit, single-channel (binary) source image.
                    This process is generally intended for images with the same width and height.
        """
        if (img := cv2.imread(file_path, cv2.IMREAD_UNCHANGED)) is None:
            raise FileNotFoundError(f'Not found or could not be read: {file_path}')

        if img.ndim != 2:
            raise ValueError(f"Requires a grayscale noise image. shape: {img.shape}")

        if img.dtype == "uint16":
            img = (img >> 8).astype("uint8")

        bit = 8 if bit8 else 16
        heightmap = self.generate_heightmap(img)
        heightmap = self.clip_array(heightmap, bit)
        self.write_to_file(heightmap, bit, parent, with_suffix, extension)

    def write_to_file(self, heightmap, bit, parent, with_suffix, extension):
        filename = f'{bit}bit_{self.__class__.__name__}'
        output_image.output(heightmap, filename, parent, with_suffix, extension)


class VoronoiNoiseRockPillars(RockPillarsHeightmap):

    def __init__(self, grid, size, t, cliff_area, random_height, blur, exclude_edge_cells):
        super().__init__(cliff_area, random_height, blur, exclude_edge_cells)
        self.grid = grid
        self.size = size
        self.t = t

    def output_heightmap(self, parent='.', with_suffix=True, extension='png', bit8=True):
        bit = 8 if bit8 else 16
        heightmap = self.create_heightmap(bit)
        self.write_to_file(heightmap, bit, parent, with_suffix, extension)


class VoronoiRockPillarsHeightmap(VoronoiNoiseRockPillars):
    """A class for generating terrain heightmap like rock pillars dynamically from voronoi noise.
        Args:
            grid (int): The number of grids. Default is 4.
            size (int): Image size. Images with the same width and height are generated. Default is 256.
            t (None or float):
                The elapsed time. If set to None, a random value is generated. Default is None.
                If specify a float value, the same hegihtmap will be generated each time the program runs.
            cliff_area (float):
                Specify how far from the edge of the Voronoi cell the gradient should extend.
                Default is 0.4.
            random_height (bool):
                If True, the height of the cliff tops will all be set to 1.0.
                Otherwise, they will be selected randomly from RockPillarsHeightmap.rock_heights. Default is False.
            blur (int):
                Specify None or an odd number. Negative values are not allowed. Default is 11.
            exclude_edge_cells (bool):
                If True, exclude Voronoi regions that are cut off at the edges of the image. Default is True
    """

    def __init__(self, grid=4, size=256, t=None, cliff_area=0.4,
                 random_height=False, blur=11, exclude_edge_cells=True):
        super().__init__(grid, size, t, cliff_area, random_height, blur, exclude_edge_cells)

    def create_heightmap(self, bit=None):
        voronoi = VoronoiEdges(self.grid)
        t = self.t if self.t is not None else random.uniform(0, 1000)
        noise_arr = voronoi.noise2(self.size, cell=1.0, edge=0.0, t=t)
        noise_arr = self.clip_array(noise_arr, bit=8)
        heightmap_arr = self.generate_heightmap(noise_arr)

        if bit is None:
            return heightmap_arr

        return self.clip_array(heightmap_arr, bit)


class RoundedVoronoiRockPillarsHeightmap(VoronoiNoiseRockPillars):
    """A class for generating terrain heightmap like rock pillars dynamically from rounded voronoi noise.
        Args:
            grid (int): The number of grids. Default is 4.
            size (int): Image size. Images with the same width and height are generated. Default is 256.
            tp (int): The smaller the value, the more rounded the corners of the Voronoi cells become.
            t (None or float):
                The elapsed time. If set to None, a random value is generated. Default is None.
                If specify a float value, the same hegihtmap will be generated each time the program runs.
            cliff_area (float):
                Specify how far from the edge of the Voronoi cell the gradient should extend.
                Default is 0.4.
            random_height (bool):
                If True, the height of the cliff tops will all be set to 1.0.
                Otherwise, they will be selected randomly from RockPillarsHeightmap.rock_heights. Default is False.
            blur (int):
                Specify None or an odd number. Negative values are not allowed. Default is 11.
            exclude_edge_cells (bool):
                If True, exclude Voronoi regions that are cut off at the edges of the image. Default is True
    """

    def __init__(self, grid=4, size=256, tp=10, t=None, cliff_area=0.4,
                 random_height=False, blur=11, exclude_edge_cells=True):
        super().__init__(grid, size, t, cliff_area, random_height, blur, exclude_edge_cells)
        self.tp = tp

    def create_heightmap(self, clipping=None):
        voronoi = VoronoiRoundEdges(self.grid)
        t = self.t if self.t is not None else random.uniform(0, 1000)
        noise_arr = voronoi.noise2(self.size, cell=1.0, edge=0.0, tp=self.tp, t=t)
        noise_arr = self.clip_array(noise_arr, bit=8)
        heightmap_arr = self.generate_heightmap(noise_arr)

        if clipping is None:
            return heightmap_arr

        return self.clip_array(heightmap_arr, clipping)