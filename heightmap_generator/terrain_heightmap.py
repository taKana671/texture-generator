import random

import cv2
import numpy as np

from noise import VoronoiEdges, VoronoiRoundEdges
from texture_generator.utils import output_image
from texture_generator.utils.noise_process_mixin import NoiseProcessMixin


class VoronoiRockPillars(NoiseProcessMixin):

    """A class for generating terrain heightmap like rock pillars.
        Args:

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
            return (v, v)
        return v

    def create_from_file(self, file_path,
                         parent='.', with_suffix=True, bit8=True):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.create_heightmap(img, parent, with_suffix, bit8)

    def create_from_voronoi(self, grid=4, size=256, t=None,
                            parent='.', with_suffix=True, bit8=True):
        voronoi = VoronoiEdges(grid)
        arr = voronoi.noise2(size, cell=1.0, edge=0.0, t=t)
        arr = self.clip_array(arr, bit8=True)
        self.create_heightmap(arr, parent, with_suffix, bit8)

    def create_from_rounded_voronoi(self, grid=4, size=256, tp=10, t=None,
                                    parent='.', with_suffix=True, bit8=True):
        voronoi = VoronoiRoundEdges(grid)
        arr = voronoi.noise2(size, cell=1.0, edge=0.0, tp=tp, t=t)
        arr = self.clip_array(arr, bit8=True)
        self.create_heightmap(arr, parent, with_suffix, bit8)

    def create_heightmap(self, arr, parent, with_suffix, bit8):
        h, w = arr.shape[:2]
        heightmap = np.zeros((h, w), dtype=np.float32)

        _, thresh = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
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
                max_val = random.choice(VoronoiRockPillars.rock_heights)

            cushion = np.clip(cell_dists / flat_threshold, 0.0, max_val)
            heightmap[y_indices, x_indices] = cushion

        if self.ksize is not None:
            heightmap = cv2.GaussianBlur(heightmap, self.ksize, 0)

        heightmap = self.clip_array(heightmap, bit8)
        prefix = '8bit' if bit8 else '16bit'
        output_image.output(heightmap, f'{prefix}_rock_pillars_heightmap', parent, with_suffix)



        # # 0.0〜1.0 のデータを 0〜255 の白黒画像（8bit）に変換して保存
        # final_heightmap = (heightmap * 255.0).astype(np.uint8)


