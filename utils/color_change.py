import cv2
import numpy as np

from .output_image import output


class ChangeColor:

    @staticmethod
    def hsv_matching(src_img, ref_img, match_rate=1.0):
        """Adjusts the color of a source image to match that of a reference image.
            Args:
                src_img (np.ndarray): Texture images whose colors to be changed (BGR)
                ref_img (np.ndarray): Texture images to use as a reference (BGR)
                match_rate (float):
                    Scale. A value of 1.0 means the image is fully adjusted to the standard,
                    while 0.0 leaves it unchanged. (0.0-1.0)

            Returns:
                np.ndarray: Texture image with adjusted color tones (BGR)
                The image size is the same as src_img.
        """
        # Convert to HSV space.
        hsv_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Separated by channel.
        h_src, s_src, v_src = cv2.split(hsv_src)
        h_ref, s_ref, v_ref = cv2.split(hsv_ref)

        # Calculate the mean and standard deviation for each channel.
        mean_h_src, _ = np.mean(h_src), np.std(h_src)
        mean_s_src, std_s_src = np.mean(s_src), np.std(s_src)
        mean_v_src, std_v_src = np.mean(v_src), np.std(v_src)

        mean_h_ref, _ = np.mean(h_ref), np.std(h_ref)
        mean_s_ref, std_s_ref = np.mean(s_ref), np.std(s_ref)
        mean_v_ref, std_v_ref = np.mean(v_ref), np.std(v_ref)

        # A very small value to prevent division by zero.
        epsilon = 1e-6

        # Adjusting Saturation (S) and Brightness (V).
        s_matched = ((s_src - mean_s_src) * (std_s_ref / (std_s_src + epsilon))) + mean_s_ref
        v_matched = ((v_src - mean_v_src) * (std_v_ref / (std_v_src + epsilon))) + mean_v_ref

        # Adjusting Hue (H).
        diff_h = mean_h_ref - mean_h_src
        h_matched = (h_src + diff_h) % 180

        # Blend with the original image based on the `match_rate`.
        h_final = (h_matched * match_rate + h_src * (1.0 - match_rate)) % 180
        s_final = s_matched * match_rate + s_src * (1.0 - match_rate)
        v_final = v_matched * match_rate + v_src * (1.0 - match_rate)

        # Clip each channel to the appropriate range.
        h_final = np.clip(h_final, 0, 179)
        s_final = np.clip(s_final, 0, 255)
        v_final = np.clip(v_final, 0, 255)

        # Convert the type back to uint8 and concatenate.
        hsv_matched = cv2.merge([
            h_final.astype(np.uint8),
            s_final.astype(np.uint8),
            v_final.astype(np.uint8)
        ])

        # Return to BGR and output.
        return cv2.cvtColor(hsv_matched, cv2.COLOR_HSV2BGR)

    @staticmethod
    def hsv_robust_matching(src_img, ref_img, match_rate=1.0):
        # Convert to HSV space.
        hsv_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV).astype(np.float32)

        h_src, s_src, v_src = cv2.split(hsv_src)
        h_ref, s_ref, v_ref = cv2.split(hsv_ref)

        # Retrieve the representative color (average) from the reference image.
        mean_h_ref = np.mean(h_ref)
        mean_s_ref = np.mean(s_ref)
        mean_v_ref = np.mean(v_ref)

        # Hue (H) and Saturation (S) are based on the target color, disregarding the original distribution.
        h_target = np.full_like(h_src, mean_h_ref)
        s_target = np.full_like(s_src, mean_s_ref)

        # Luminance (V) adjusts only the average brightness of the entire image.
        mean_v_src = np.mean(v_src)
        v_target = v_src + (mean_v_ref - mean_v_src)

        # Blend based on the `match_rate`.
        h_final = h_target * match_rate + h_src * (1.0 - match_rate)
        s_final = s_target * match_rate + s_src * (1.0 - match_rate)
        v_final = v_target * match_rate + v_src * (1.0 - match_rate)

        # Clip each channel to the appropriate range.
        h_final = np.clip(h_final % 180, 0, 179)
        s_final = np.clip(s_final, 0, 255)
        v_final = np.clip(v_final, 0, 255)

        hsv_matched = cv2.merge([
            h_final.astype(np.uint8),
            s_final.astype(np.uint8),
            v_final.astype(np.uint8)
        ])

        return cv2.cvtColor(hsv_matched, cv2.COLOR_HSV2BGR)

    @staticmethod
    def lab_matching(src_img, ref_img, match_rate=1.0):
        """Uses the LAB color space (Reinhard method) to adjust the color of a source image to match that of a reference image.
            Args:
                src_img (np.ndarray): Texture images whose colors to be changed (BGR)
                ref_img (np.ndarray): Texture images to use as a reference (BGR)
                match_rate (float):
                    Scale. A value of 1.0 means the image is fully adjusted to the standard,
                    while 0.0 leaves it unchanged. (0.0-1.0)

            Returns:
                np.ndarray:
                    Texture image with adjusted color tones (BGR)
                    The image size is the same as src_img.
        """

        # Convert to LAB space.
        lab_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2Lab).astype(np.float32)

        # Separated into individual channels (L: Luminance, a: Green to Red, b: Blue to Yellow)
        l_src, a_src, b_src = cv2.split(lab_src)
        l_ref, a_ref, b_ref = cv2.split(lab_ref)

        # Calculate the mean and standard deviation for each channel.
        mean_l_src, std_l_src = np.mean(l_src), np.std(l_src)
        mean_a_src, std_a_src = np.mean(a_src), np.std(a_src)
        mean_b_src, std_b_src = np.mean(b_src), np.std(b_src)

        mean_l_ref, std_l_ref = np.mean(l_ref), np.std(l_ref)
        mean_a_ref, std_a_ref = np.mean(a_ref), np.std(a_ref)
        mean_b_ref, std_b_ref = np.mean(b_ref), np.std(b_ref)

        # A very small value to prevent division by zero.
        epsilon = 1e-6

        # Subtract the original mean, multiply by the ratio of the target standard deviation
        # to the original standard deviation, and add the target mean.
        l_matched = ((l_src - mean_l_src) * (std_l_ref / (std_l_src + epsilon))) + mean_l_ref
        a_matched = ((a_src - mean_a_src) * (std_a_ref / (std_a_src + epsilon))) + mean_a_ref
        b_matched = ((b_src - mean_b_src) * (std_b_ref / (std_b_src + epsilon))) + mean_b_ref

        # Blend with the original image based on the `match_rate`.
        l_final = l_matched * match_rate + l_src * (1.0 - match_rate)
        a_final = a_matched * match_rate + a_src * (1.0 - match_rate)
        b_final = b_matched * match_rate + b_src * (1.0 - match_rate)

        # Clip the values of each channel to the appropriate range (0–255 for OpenCV's LAB).
        l_final = np.clip(l_final, 0, 255)
        a_final = np.clip(a_final, 0, 255)
        b_final = np.clip(b_final, 0, 255)

        # Convert the type back to uint8 and concatenate.
        lab_matched = cv2.merge([
            l_final.astype(np.uint8),
            a_final.astype(np.uint8),
            b_final.astype(np.uint8)
        ])

        # Return to BGR and output.
        return cv2.cvtColor(lab_matched, cv2.COLOR_Lab2BGR)


def match_texcolor_hsv(src_img_path,
                       ref_img_path,
                       match_rate=1.0,
                       parent='.',
                       with_suffix=True,
                       file_extension='png'
                       ):
    src_img = cv2.imread(src_img_path)
    ref_img = cv2.imread(ref_img_path)

    arr = ChangeColor.hsv_matching(src_img, ref_img, match_rate)
    output(arr, 'matched_texture_hsv', parent, with_suffix, file_extension)


def match_texcolor_hsv_robust(src_img_path,
                              ref_img_path,
                              match_rate=1.0,
                              parent='.',
                              with_suffix=True,
                              file_extension='png'
                              ):
    src_img = cv2.imread(src_img_path)
    ref_img = cv2.imread(ref_img_path)

    arr = ChangeColor.hsv_robust_matching(src_img, ref_img, match_rate)
    output(arr, 'matched_texture_hsv_robust', parent, with_suffix, file_extension)


def match_texcolor_lab(src_img_path,
                       ref_img_path,
                       match_rate=1.0,
                       parent='.',
                       with_suffix=True,
                       file_extension='png'
                       ):
    src_img = cv2.imread(src_img_path)
    ref_img = cv2.imread(ref_img_path)

    arr = ChangeColor.lab_matching(src_img, ref_img, match_rate)
    output(arr, 'matched_texture_lab', parent, with_suffix, file_extension)


# s_matched = ((s_src - mean_s_src) * (std_s_ref / (std_s_src + epsilon))) + mean_s_ref
# 1 (s_src - mean_s_src):
#     元画像の各画素の値から、元画像全体の平均値を引く。
#     これにより、全体の明るさや色味がリセットされ、平均より高い場所（プラス）と低い場所（マイナス）に分かれる。
#    （ベースの色味を一旦消し去り、純粋なテクスチャの「凸凹の模様」だけを浮き彫りにする）
# 2  * (std_s_ref / std_s_src)
#     基準（ターゲット）画像の標準偏差（ばらつき）を、元画像の標準偏差で割り算し、それを掛ける。
#     基準画像が「すごくクッキリした（std大）」画像で、元画像が「ぼんやりした（std小）」画像なら、この比率は 2.0 や 3.0 といった大きな値になる。
# 　　これを掛け算することで、元画像の凸凹模様を、基準画像のクッキリ度（または滑らかさ）に合わせて拡大・縮小する。
# 3 + mean_s_ref
#     最後に、基準（ターゲット）画像全体の平均値を足す。
#     1〜2で「基準画像と同じメリハリ度になった純粋な模様」に対して、基準画像のベースとなる色味（平均値）を乗せる（スライドさせる）。
#     これにより、元画像の模様を保ったまま、見た目の色や明るさを基準画像に寄せる。
