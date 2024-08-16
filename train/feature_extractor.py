import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import regionprops
from skimage.filters import gabor
from mahotas.features import haralick, zernike_moments


class PotholeFeatureExtractor:
    def __init__(self):
        pass

    def pixel_to_mm_conversion(self, annotations, image_width, image_height):
        l1_annotation = next((ann for ann in annotations if ann["class"] == "L1"), None)

        if l1_annotation:
            width = l1_annotation["width"]
            height = l1_annotation["height"]
            pixel_width = width * image_width
            pixel_height = height * image_height

            # L1 true length is 500mm
            mm_per_pixel_width = 500 / pixel_width
            mm_per_pixel_height = 500 / pixel_height

            # Use the average of width and height conversion factors
            mm_per_pixel = (mm_per_pixel_width + mm_per_pixel_height) / 2

            return mm_per_pixel
        else:
            return None

    def calculate_fractal_dimension(self, Z):
        assert len(Z.shape) == 2

        def box_count(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k),
                axis=1,
            )
            return len(np.where((S > 0) & (S < k * k))[0])

        Z = (Z < np.mean(Z)) * 1
        p = min(Z.shape)
        n = 2 ** np.floor(np.log(p) / np.log(2))
        n = int(np.log(n) / np.log(2))
        sizes = 2 ** np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(box_count(Z, size))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def extract(self, image, annotations):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width, _ = image.shape

        mm_per_pixel = self.pixel_to_mm_conversion(
            annotations, image_width, image_height
        )

        features = {}

        pothole_annotation = next(
            (ann for ann in annotations if ann["class"] == "pothole"), None
        )
        if pothole_annotation:
            x_center = pothole_annotation["x"]
            y_center = pothole_annotation["y"]
            width = pothole_annotation["width"]
            height = pothole_annotation["height"]

            x_min = int((x_center - width / 2) * image_width)
            x_max = int((x_center + width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            y_max = int((y_center + height / 2) * image_height)

            roi = image_rgb[y_min:y_max, x_min:x_max]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            features["pothole_width_pixels"] = x_max - x_min
            features["pothole_height_pixels"] = y_max - y_min
            features["pothole_area_pixels"] = (
                features["pothole_width_pixels"] * features["pothole_height_pixels"]
            )
            features["aspect_ratio"] = (
                features["pothole_width_pixels"] / features["pothole_height_pixels"]
            )
            features["relative_size"] = features["pothole_area_pixels"] / (
                image_width * image_height
            )

            if mm_per_pixel:
                features["pothole_width_mm"] = (
                    features["pothole_width_pixels"] * mm_per_pixel
                )
                features["pothole_height_mm"] = (
                    features["pothole_height_pixels"] * mm_per_pixel
                )
                features["pothole_area_mm2"] = features["pothole_area_pixels"] * (
                    mm_per_pixel**2
                )

            for i, color in ["r", "g", "b"]:
                features[f"avg_color_{color}"] = np.mean(roi[:, :, i])
                features[f"std_color_{color}"] = np.std(roi[:, :, i])
                features[f"max_color_{color}"] = np.max(roi[:, :, i])
                features[f"min_color_{color}"] = np.min(roi[:, :, i])

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            for i, color in ["h", "s", "v"]:
                features[f"avg_hsv_{color}"] = np.mean(roi_hsv[:, :, i])
                features[f"std_hsv_{color}"] = np.std(roi_hsv[:, :, i])

            features["compactness"] = (4 * np.pi * features["pothole_area_pixels"]) / (
                features["pothole_width_pixels"] ** 2
                + features["pothole_height_pixels"] ** 2
            )
            features["perimeter_pixels"] = 2 * (
                features["pothole_width_pixels"] + features["pothole_height_pixels"]
            )
            if mm_per_pixel:
                features["perimeter_mm"] = features["perimeter_pixels"] * mm_per_pixel

            f_transform = np.fft.fft2(roi_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            features["fourier_mean"] = np.mean(magnitude_spectrum)
            features["fourier_std"] = np.std(magnitude_spectrum)
            features["fourier_max"] = np.max(magnitude_spectrum)

            edges = cv2.Canny(roi_gray, 100, 200)
            features["edge_density"] = np.sum(edges) / features["pothole_area_pixels"]

            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            features["hist_mean"] = np.mean(hist)
            features["hist_std"] = np.std(hist)
            features["hist_skewness"] = skew(hist.flatten())
            features["hist_kurtosis"] = kurtosis(hist.flatten())
            features["hist_energy"] = np.sum(hist**2)
            features["hist_entropy"] = -np.sum(hist * np.log2(hist + 1e-7))

            glcm = graycomatrix(
                roi_gray,
                [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                symmetric=True,
                normed=True,
            )
            for prop in [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
            ]:
                features[f"glcm_{prop}"] = graycoprops(glcm, prop)[0, 0]

            roi_binary = cv2.threshold(
                roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            props = regionprops(roi_binary)[0]
            features["eccentricity"] = props.eccentricity
            features["solidity"] = props.solidity
            features["extent"] = props.extent
            features["euler_number"] = props.euler_number
            features["equivalent_diameter"] = props.equivalent_diameter
            features["orientation"] = props.orientation

            gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            features["gradient_mean_magnitude"] = np.mean(mag)
            features["gradient_std_magnitude"] = np.std(mag)
            features["gradient_mean_angle"] = np.mean(ang)
            features["gradient_std_angle"] = np.std(ang)

            for radius in [1, 2, 3]:
                n_points = 8 * radius
                lbp = local_binary_pattern(roi_gray, n_points, radius, method="uniform")
                features[f"lbp_mean_r{radius}"] = np.mean(lbp)
                features[f"lbp_std_r{radius}"] = np.std(lbp)

            haralick_features = haralick(roi_gray).mean(axis=0)
            for i, value in enumerate(haralick_features):
                features[f"haralick_{i}"] = value

            moments = cv2.moments(roi_gray)
            huMoments = cv2.HuMoments(moments)
            for i in range(7):
                features[f"hu_moment_{i}"] = -np.sign(huMoments[i]) * np.log10(
                    np.abs(huMoments[i])
                )

            zernike = zernike_moments(roi_gray, radius=min(roi_gray.shape) // 2)
            for i, z in enumerate(zernike):
                features[f"zernike_{i}"] = z

            # Gabor filter features
            num_angles = 4
            frequencies = [0.1, 0.5]
            for theta in range(num_angles):
                theta = theta / num_angles * np.pi
                for frequency in frequencies:
                    filt_real, filt_imag = gabor(
                        roi_gray, frequency=frequency, theta=theta
                    )
                    features[f"gabor_mean_f{frequency:.1f}_t{theta:.2f}"] = np.mean(
                        filt_real
                    )
                    features[f"gabor_std_f{frequency:.1f}_t{theta:.2f}"] = np.std(
                        filt_real
                    )

            # HOG features
            hog_features, _ = hog(
                roi_gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
            )

            # Summarize HOG features
            features["hog_mean"] = np.mean(hog_features)
            features["hog_std"] = np.std(hog_features)
            features["hog_max"] = np.max(hog_features)
            features["hog_min"] = np.min(hog_features)

            # Compute HOG features for specific regions
            h, w = roi_gray.shape
            center_roi = roi_gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            edge_roi = roi_gray - cv2.erode(roi_gray, None)

            for region, name in [(center_roi, "center"), (edge_roi, "edge")]:
                region_hog, _ = hog(
                    region,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                )
                features[f"hog_{name}_mean"] = np.mean(region_hog)
                features[f"hog_{name}_std"] = np.std(region_hog)

            # Compute histogram of HOG features
            hist, _ = np.histogram(hog_features, bins=10, range=(0, 1))
            for i, count in enumerate(hist):
                features[f"hog_hist_bin_{i}"] = count

            # Blob detection
            params = cv2.SimpleBlobDetector_Params()
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(roi_gray)
            features["num_blobs"] = len(keypoints)
            if keypoints:
                features["avg_blob_size"] = np.mean([k.size for k in keypoints])
                features["std_blob_size"] = np.std([k.size for k in keypoints])

            # Contour features
            contours, _ = cv2.findContours(
                roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                features["contour_area"] = cv2.contourArea(main_contour)
                features["contour_perimeter"] = cv2.arcLength(main_contour, True)
                features["contour_circularity"] = (
                    4
                    * np.pi
                    * features["contour_area"]
                    / (features["contour_perimeter"] ** 2)
                )

            # Fractal dimension (box counting)
            features["fractal_dimension"] = self.calculate_fractal_dimension(roi_gray)

        return features


if __name__ == "__main__":
    extractor = PotholeFeatureExtractor()

    example_input = {
        "image": cv2.imread("path_to_image.jpg"),  # Replace with actual image path
        "annotations": [
            {"class": "pothole", "x": 0.5, "y": 0.5, "width": 0.3, "height": 0.3},
            {"class": "L1", "x": 0.7, "y": 0.7, "width": 0.2, "height": 0.2},
        ],
    }

    features = extractor.extract(example_input["image"], example_input["annotations"])
    print("Extracted features:")
    for key, value in features.items():
        print(f"{key}: {value}")
