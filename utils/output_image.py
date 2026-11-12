import pathlib
from datetime import datetime

import cv2


def make_dir_name(name, parent='.'):
    now = datetime.now()
    path = pathlib.Path(f'{parent}/{name}_{now.strftime("%Y%m%d%H%M%S")}')
    return path


def output(arr, stem, parent='.', with_suffix=True, file_extension='png'):
    """Output an image file.
       If the folder specified as the output destination does not exist, it will be created.
        Args:
            arr (numpy.ndarray): An image to be output.
            stem (str): File name without extension.
            parent (str): Folders
            with_suffix (bool): If True, datetime will be included in the file name.
            file_extension (str): png or jpg
    """
    if parent != '.':
        pathlib.Path(parent).mkdir(parents=True, exist_ok=True)

    if with_suffix:
        now = datetime.now()
        stem = f'{stem}_{now.strftime("%Y%m%d%H%M%S")}'

    file_path = f'{parent}/{stem}.{file_extension}'
    cv2.imwrite(file_path, arr)