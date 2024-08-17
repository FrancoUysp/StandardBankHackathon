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
        # Find the L1 annotation
        for ann in annotations:
            if ann["class"] == "L1":
                # Calculate mm per pixel directly
                mm_per_pixel = 500 / (
                    (ann["width"] * image_width + ann["height"] * image_height) / 2
                )
                return mm_per_pixel

        # Return None if no L1 annotation is found
        return None

    def calculate_fractal_dimension(self, Z):
        assert len(Z.shape) == 2

        def box_count(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k),
                axis=1,
            )
            return np.sum((S > 0) & (S < k * k))

        Z_mean = np.mean(Z)
        Z = (Z < Z_mean).astype(int)
        p = min(Z.shape)
        n = int(np.log2(p))
        sizes = 2 ** np.arange(n, 1, -1)

        counts = np.array([box_count(Z, size) for size in sizes])
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

        return -coeffs[0]

    def extract(self, image, annotations):
        # Precompute values and convert image formats only once where necessary
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width, _ = image.shape

        mm_per_pixel = self.pixel_to_mm_conversion(
            annotations, image_width, image_height
        )

        features = {}

        # Find pothole annotation early and calculate bounding box
        pothole_annotation = next(
            (ann for ann in annotations if ann["class"] == "pothole"), None
        )
        if pothole_annotation:
            x_center, y_center = pothole_annotation["x"], pothole_annotation["y"]
            width, height = pothole_annotation["width"], pothole_annotation["height"]

            x_min = int((x_center - width / 2) * image_width)
            x_max = int((x_center + width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            y_max = int((y_center + height / 2) * image_height)

            roi = image_rgb[y_min:y_max, x_min:x_max]
            roi_gray = image_gray[y_min:y_max, x_min:x_max]  # Use precomputed grayscale

            # Precompute pixel dimensions and area
            pothole_width_pixels = x_max - x_min
            pothole_height_pixels = y_max - y_min
            pothole_area_pixels = pothole_width_pixels * pothole_height_pixels

            features["pothole_width_pixels"] = pothole_width_pixels
            features["pothole_height_pixels"] = pothole_height_pixels
            features["pothole_area_pixels"] = pothole_area_pixels
            features["aspect_ratio"] = pothole_width_pixels / pothole_height_pixels
            features["relative_size"] = pothole_area_pixels / (
                image_width * image_height
            )

            if mm_per_pixel:
                features["pothole_width_mm"] = pothole_width_pixels * mm_per_pixel
                features["pothole_height_mm"] = pothole_height_pixels * mm_per_pixel
                features["pothole_area_mm2"] = pothole_area_pixels * (mm_per_pixel**2)

            # Optimize RGB and HSV feature extraction using precomputed regions
            for i, color in enumerate(["r", "g", "b"]):
                channel = roi[:, :, i]
                features[f"avg_color_{color}"] = np.mean(channel)
                features[f"std_color_{color}"] = np.std(channel)
                features[f"max_color_{color}"] = np.max(channel)
                features[f"min_color_{color}"] = np.min(channel)

            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            for i, color in enumerate(["h", "s", "v"]):
                channel = roi_hsv[:, :, i]
                features[f"avg_hsv_{color}"] = np.mean(channel)
                features[f"std_hsv_{color}"] = np.std(channel)

            # Combine compactness, perimeter, and Fourier transform calculations
            perimeter_pixels = 2 * (pothole_width_pixels + pothole_height_pixels)
            features["compactness"] = (4 * np.pi * pothole_area_pixels) / (
                pothole_width_pixels**2 + pothole_height_pixels**2
            )
            features["perimeter_pixels"] = perimeter_pixels
            if mm_per_pixel:
                features["perimeter_mm"] = perimeter_pixels * mm_per_pixel

            f_transform = np.fft.fft2(roi_gray)
            magnitude_spectrum = 20 * np.log(np.abs(np.fft.fftshift(f_transform)))
            features["fourier_mean"] = np.mean(magnitude_spectrum)
            features["fourier_std"] = np.std(magnitude_spectrum)
            features["fourier_max"] = np.max(magnitude_spectrum)

            # Optimize edge detection and histogram calculations
            edges = cv2.Canny(roi_gray, 100, 200)
            features["edge_density"] = np.sum(edges) / pothole_area_pixels

            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            hist_flat = hist.flatten()
            features["hist_mean"] = np.mean(hist_flat)
            features["hist_std"] = np.std(hist_flat)
            features["hist_skewness"] = skew(hist_flat)
            features["hist_kurtosis"] = kurtosis(hist_flat)
            features["hist_energy"] = np.sum(hist_flat**2)
            features["hist_entropy"] = -np.sum(hist_flat * np.log2(hist_flat + 1e-7))

            # GLCM features optimization
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

            # Blob detection, contour, and other features
            roi_binary = cv2.threshold(
                roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            props = regionprops(roi_binary)[0]
            features.update(
                {
                    "eccentricity": props.eccentricity,
                    "solidity": props.solidity,
                    "extent": props.extent,
                    "euler_number": props.euler_number,
                    "equivalent_diameter": props.equivalent_diameter,
                    "orientation": props.orientation,
                }
            )

            # Sobel gradient calculations
            gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            features["gradient_mean_magnitude"] = np.mean(mag)
            features["gradient_std_magnitude"] = np.std(mag)
            features["gradient_mean_angle"] = np.mean(ang)
            features["gradient_std_angle"] = np.std(ang)

            # Local binary pattern and Haralick features
            for radius in [1, 2, 3]:
                lbp = local_binary_pattern(
                    roi_gray, 8 * radius, radius, method="uniform"
                )
                features[f"lbp_mean_r{radius}"] = np.mean(lbp)
                features[f"lbp_std_r{radius}"] = np.std(lbp)

            haralick_features = haralick(roi_gray).mean(axis=0)
            for i, value in enumerate(haralick_features):
                features[f"haralick_{i}"] = value

            # Hu moments
            moments = cv2.moments(roi_gray)
            hu_moments = cv2.HuMoments(moments)
            for i in range(7):
                features[f"hu_moment_{i}"] = -np.sign(hu_moments[i]) * np.log10(
                    np.abs(hu_moments[i])
                )

            # Zernike moments
            zernike = zernike_moments(roi_gray, radius=min(roi_gray.shape) // 2)
            for i, z in enumerate(zernike):
                features[f"zernike_{i}"] = z

            # Gabor filter features
            num_angles = 4
            frequencies = [0.1, 0.5]
            for theta in range(num_angles):
                theta_rad = theta * np.pi / num_angles
                for frequency in frequencies:
                    filt_real, filt_imag = gabor(
                        roi_gray, frequency=frequency, theta=theta_rad
                    )
                    features[f"gabor_mean_f{frequency:.1f}_t{theta_rad:.2f}"] = np.mean(
                        filt_real
                    )
                    features[f"gabor_std_f{frequency:.1f}_t{theta_rad:.2f}"] = np.std(
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
            features["hog_mean"] = np.mean(hog_features)
            features["hog_std"] = np.std(hog_features)
            features["hog_max"] = np.max(hog_features)
            features["hog_min"] = np.min(hog_features)

            # Histogram of HOG features
            hist, _ = np.histogram(hog_features, bins=10, range=(0, 1))
            for i, count in enumerate(hist):
                features[f"hog_hist_bin_{i}"] = count

            # Blob detection
            params = cv2.SimpleBlobDetector_Params()
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(roi_gray)
            features["num_blobs"] = len(keypoints)
            if keypoints:
                blob_sizes = [k.size for k in keypoints]
                features["avg_blob_size"] = np.mean(blob_sizes)
                features["std_blob_size"] = np.std(blob_sizes)

            contours, _ = cv2.findContours(
                roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(main_contour)
                contour_perimeter = cv2.arcLength(main_contour, True)
                features["contour_area"] = contour_area
                features["contour_perimeter"] = contour_perimeter
                features["contour_circularity"] = (4 * np.pi * contour_area) / (
                    contour_perimeter**2
                )

            # Fractal dimension (box counting)
            features["fractal_dimension"] = self.calculate_fractal_dimension(roi_gray)

        return features


if __name__ == "__main__":
    extractor = PotholeFeatureExtractor()

    example_input = {
        "image": cv2.imread("data/images/p1000.jpg"),  # Replace with actual image path
        "annotations": [
            {
                "class": "pothole",
                "x": 0.634640,
                "y": 0.57,
                "width": 0.59,
                "height": 0.76,
            },
            {"class": "L1", "x": 0.42, "y": 0.42, "width": 0.24, "height": 0.044},
        ],
    }

    features = extractor.extract(example_input["image"], example_input["annotations"])
    print("Extracted features:")
    for key, value in features.items():
        print(f"{key}: {value}")
