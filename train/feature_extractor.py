import cv2
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import regionprops
from skimage.filters import gabor
from mahotas.features import haralick, zernike_moments
from sklearn.preprocessing import normalize

class PotholeFeatureExtractor:
    def __init__(self):
        pass

    def read_yolo_annotation(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            annotations.append((label, x_center, y_center, width, height))
        return annotations

    def pixel_to_mm_conversion(self, annotations, image_width, image_height):
        l1_annotation = next((ann for ann in annotations if ann[0] == 1), None)
        
        if l1_annotation:
            _, _, _, width, height = l1_annotation
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
        # Only for 2d image
        assert(len(Z.shape) == 2)

        def box_count(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])

        # Transform Z into a binary array
        Z = (Z < np.mean(Z)) * 1

        # Minimal dimension of image
        p = min(Z.shape)

        # Greatest power of 2 less than or equal to p
        n = 2**np.floor(np.log(p)/np.log(2))

        # Extract the exponent
        n = int(np.log(n)/np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)

        # Actual box counting with decreasing size
        counts = []
        for size in sizes:
            counts.append(box_count(Z, size))

        # Fit the successive log(sizes) with log(counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def extract_features(self, image_path, annotation_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width, _ = image.shape
        
        # Read annotations
        annotations = self.read_yolo_annotation(annotation_path)
        
        # Calculate mm per pixel
        mm_per_pixel = self.pixel_to_mm_conversion(annotations, image_width, image_height)
        
        features = {}
        
        # Extract features for the pothole (assuming class 0 is the pothole)
        pothole_annotation = next((ann for ann in annotations if ann[0] == 0), None)
        if pothole_annotation:
            _, x_center, y_center, width, height = pothole_annotation
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - width / 2) * image_width)
            x_max = int((x_center + width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            y_max = int((y_center + height / 2) * image_height)
            
            # Extract ROI
            roi = image_rgb[y_min:y_max, x_min:x_max]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            
            # Basic geometric features
            features['pothole_width_pixels'] = x_max - x_min
            features['pothole_height_pixels'] = y_max - y_min
            features['pothole_area_pixels'] = features['pothole_width_pixels'] * features['pothole_height_pixels']
            features['aspect_ratio'] = features['pothole_width_pixels'] / features['pothole_height_pixels']
            features['relative_size'] = features['pothole_area_pixels'] / (image_width * image_height)
            
            if mm_per_pixel:
                features['pothole_width_mm'] = features['pothole_width_pixels'] * mm_per_pixel
                features['pothole_height_mm'] = features['pothole_height_pixels'] * mm_per_pixel
                features['pothole_area_mm2'] = features['pothole_area_pixels'] * (mm_per_pixel ** 2)
            
            # Color features
            for i, color in enumerate(['r', 'g', 'b']):
                features[f'avg_color_{color}'] = np.mean(roi[:,:,i])
                features[f'std_color_{color}'] = np.std(roi[:,:,i])
                features[f'max_color_{color}'] = np.max(roi[:,:,i])
                features[f'min_color_{color}'] = np.min(roi[:,:,i])
            
            # HSV color space features
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            for i, color in enumerate(['h', 's', 'v']):
                features[f'avg_hsv_{color}'] = np.mean(roi_hsv[:,:,i])
                features[f'std_hsv_{color}'] = np.std(roi_hsv[:,:,i])
            
            # Shape features
            features['compactness'] = (4 * np.pi * features['pothole_area_pixels']) / (features['pothole_width_pixels'] ** 2 + features['pothole_height_pixels'] ** 2)
            features['perimeter_pixels'] = 2 * (features['pothole_width_pixels'] + features['pothole_height_pixels'])
            if mm_per_pixel:
                features['perimeter_mm'] = features['perimeter_pixels'] * mm_per_pixel
            
            # Fourier transform features
            f_transform = np.fft.fft2(roi_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            features['fourier_mean'] = np.mean(magnitude_spectrum)
            features['fourier_std'] = np.std(magnitude_spectrum)
            features['fourier_max'] = np.max(magnitude_spectrum)
            
            # Edge detection features
            edges = cv2.Canny(roi_gray, 100, 200)
            features['edge_density'] = np.sum(edges) / features['pothole_area_pixels']
            
            # Histogram features
            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            features['hist_mean'] = np.mean(hist)
            features['hist_std'] = np.std(hist)
            features['hist_skewness'] = skew(hist.flatten())
            features['hist_kurtosis'] = kurtosis(hist.flatten())
            features['hist_energy'] = np.sum(hist**2)
            features['hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-7))
            
            # GLCM texture features
            glcm = graycomatrix(roi_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features[f'glcm_{prop}'] = graycoprops(glcm, prop)[0, 0]
            
            # Shape descriptors
            roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            props = regionprops(roi_binary)[0]
            features['eccentricity'] = props.eccentricity
            features['solidity'] = props.solidity
            features['extent'] = props.extent
            features['euler_number'] = props.euler_number
            features['equivalent_diameter'] = props.equivalent_diameter
            features['orientation'] = props.orientation
            
            # Gradient features
            gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy)
            features['gradient_mean_magnitude'] = np.mean(mag)
            features['gradient_std_magnitude'] = np.std(mag)
            features['gradient_mean_angle'] = np.mean(ang)
            features['gradient_std_angle'] = np.std(ang)
            
            # Local Binary Pattern (LBP) features
            for radius in [1, 2, 3]:
                n_points = 8 * radius
                lbp = local_binary_pattern(roi_gray, n_points, radius, method='uniform')
                features[f'lbp_mean_r{radius}'] = np.mean(lbp)
                features[f'lbp_std_r{radius}'] = np.std(lbp)
            
            # Haralick texture features
            haralick_features = haralick(roi_gray).mean(axis=0)
            for i, value in enumerate(haralick_features):
                features[f'haralick_{i}'] = value
            
            # Hu Moments
            moments = cv2.moments(roi_gray)
            huMoments = cv2.HuMoments(moments)
            for i in range(7):
                features[f'hu_moment_{i}'] = -np.sign(huMoments[i]) * np.log10(np.abs(huMoments[i]))
            
            # Zernike moments
            zernike = zernike_moments(roi_gray, radius=min(roi_gray.shape)//2)
            for i, z in enumerate(zernike):
                features[f'zernike_{i}'] = z
            
            # Gabor filter features
            num_angles = 4
            frequencies = [0.1, 0.5]
            for theta in range(num_angles):
                theta = theta / num_angles * np.pi
                for frequency in frequencies:
                    filt_real, filt_imag = gabor(roi_gray, frequency=frequency, theta=theta)
                    features[f'gabor_mean_f{frequency:.1f}_t{theta:.2f}'] = np.mean(filt_real)
                    features[f'gabor_std_f{frequency:.1f}_t{theta:.2f}'] = np.std(filt_real)
            
            # HOG features
            hog_features, _ = hog(roi_gray, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
            
            # Summarize HOG features
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_max'] = np.max(hog_features)
            features['hog_min'] = np.min(hog_features)
            
            # Compute HOG features for specific regions
            h, w = roi_gray.shape
            center_roi = roi_gray[h//4:3*h//4, w//4:3*w//4]
            edge_roi = roi_gray - cv2.erode(roi_gray, None)
            
            for region, name in [(center_roi, 'center'), (edge_roi, 'edge')]:
                region_hog, _ = hog(region, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True)
                features[f'hog_{name}_mean'] = np.mean(region_hog)
                features[f'hog_{name}_std'] = np.std(region_hog)
            
            # Compute histogram of HOG features
            hist, _ = np.histogram(hog_features, bins=10, range=(0, 1))
            for i, count in enumerate(hist):
                features[f'hog_hist_bin_{i}'] = count
            
            # Blob detection
            params = cv2.SimpleBlobDetector_Params()
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(roi_gray)
            features['num_blobs'] = len(keypoints)
            if keypoints:
                features['avg_blob_size'] = np.mean([k.size for k in keypoints])
                features['std_blob_size'] = np.std([k.size for k in keypoints])
            
            # Contour features
            contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                features['contour_area'] = cv2.contourArea(main_contour)
                features['contour_perimeter'] = cv2.arcLength(main_contour, True)
                features['contour_circularity'] = 4 * np.pi * features['contour_area'] / (features['contour_perimeter'] ** 2)
            
            # Fractal dimension (box counting)
            features['fractal_dimension'] = self.calculate_fractal_dimension(roi_gray)

        return features

    def extract_batch(self, image_dir, annotation_dir):
        all_features = []
        problem_files = []
        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, image_file)
                annotation_file = os.path.splitext(image_file)[0] + '.txt'
                annotation_path = os.path.join(annotation_dir, annotation_file)
                
                if os.path.exists(annotation_path):
                    # Print image file being processed
                    print(f"Processing image: {image_file}")
                    try:
                        features = self.extract_features(image_path, annotation_path)
                        if features:
                            features['image_file'] = image_file
                            all_features.append(features)
                    except:
                        print(f"Error: Unable to extract features for {image_file}")
                        problem_files.append(image_file)
                else:
                    print(f"Warning: No annotation file found for {image_file}")
        
        print("Problem files:" + str(problem_files))
        return all_features

# Usage example:
if __name__ == "__main__":
    extractor = PotholeFeatureExtractor()
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Single image extraction
    # image_path = os.path.join(script_dir, 'data', 'images', 'val', 'p102.jpg')
    # annotation_path = os.path.join(script_dir, 'data', 'labels', 'val', 'p102.txt')
    
    # # Check if files exist
    # if not os.path.exists(image_path):
    #     print(f"Error: Image file not found at {image_path}")
    # elif not os.path.exists(annotation_path):
    #     print(f"Error: Annotation file not found at {annotation_path}")
    # else:
    #     print(f"Processing image: {image_path}")
    #     print(f"Using annotation: {annotation_path}")
    #     features = extractor.extract_features(image_path, annotation_path)
    #     if features:
    #         print("Extracted features:")
    #         for key, value in features.items():
    #             print(f"{key}: {value}")
    #     else:
    #         print("No features were extracted. Check if the image and annotation files are valid.")
    
    # print(features.__len__())

    # @Franco - we can use this to extract features for all images in a directory instead of one by one

    image_dir = os.path.join(script_dir, 'data', 'images', 'sup')
    annotation_dir = os.path.join(script_dir, 'data', 'labels', 'sup')
    all_features = extractor.extract_batch(image_dir, annotation_dir)
    # Write the features to a CSV file
    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(script_dir, 'data', 'sup_features.csv'), index=False)
    print(f"Extracted features for {len(all_features)} images")