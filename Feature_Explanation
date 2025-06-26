# Image Metadata Extraction Features

## Overview
This document summarizes all the features extracted from images for authenticity detection in the `image_metadata_extraction.ipynb` notebook.

## 📊 Feature Categories

### 1. Basic Image Information
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `filename` | Original filename of the image | String |
| `label` | Classification label (real/edited) | String/Integer |
| `format` | Image format (JPEG, PNG, TIFF, etc.) | String |
| `width` | Image width in pixels | Integer |
| `height` | Image height in pixels | Integer |
| `has_exif` | Whether image contains EXIF metadata | Boolean |
| `has_qtable` | Whether image has quantization table (JPEG only) | Boolean |

### 2. EXIF Metadata Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `Make` | Camera manufacturer | String/None |
| `Model` | Camera model | String/None |
| `Software` | Software used for editing | String/None |
| `DateTimeOriginal` | Original capture date/time | String/None |

### 3. Quantization Table Features (JPEG only)
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `qtable_mean` | Mean value of quantization table | Float/None |
| `qtable_std` | Standard deviation of quantization table | Float/None |
| `qtable_max` | Maximum value in quantization table | Float/None |
| `qtable_min` | Minimum value in quantization table | Float/None |

### 4. Color Entropy Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `r_entropy` | Entropy of red channel histogram | Float |
| `g_entropy` | Entropy of green channel histogram | Float |
| `b_entropy` | Entropy of blue channel histogram | Float |

### 5. Image Quality Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `blur_metric` | Laplacian variance (sharpness measure) | Float |
| `edge_density` | Proportion of edge pixels in image | Float |

### 6. Hu Moments Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `hu_1` | First Hu moment (scale invariant) | Float |
| `hu_2` | Second Hu moment (scale invariant) | Float |
| `hu_3` | Third Hu moment (scale invariant) | Float |
| `hu_4` | Fourth Hu moment (scale invariant) | Float |
| `hu_5` | Fifth Hu moment (scale invariant) | Float |
| `hu_6` | Sixth Hu moment (scale invariant) | Float |
| `hu_7` | Seventh Hu moment (scale invariant) | Float |

### 7. DCT (Discrete Cosine Transform) Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `dct_mean` | Mean of absolute DCT coefficients | Float |
| `dct_std` | Standard deviation of absolute DCT coefficients | Float |
| `dct_max` | Maximum absolute DCT coefficient | Float |
| `dct_energy` | Sum of squared absolute DCT coefficients | Float |

### 8. Noise Analysis Features
| Feature | Description | Data Type |
|---------|-------------|-----------|
| `noise_mean` | Mean of noise component | Float |
| `noise_std` | Standard deviation of noise component | Float |

## 🔧 Technical Implementation

### Supported Image Formats
- **JPEG** (.jpg, .jpeg) - Full feature extraction including EXIF and quantization tables
- **PNG** (.png) - All features except EXIF and quantization tables
- **TIFF** (.tif, .tiff) - All features except quantization tables
- **Other formats** - Basic features only

### Feature Extraction Functions

#### `extract_exif_info(image_path, img_format)`
- Extracts camera and software metadata from JPEG/TIFF images
- Returns camera make, model, software, and original date/time

#### `extract_qtable(image_path, img_format)`
- Extracts JPEG quantization table statistics
- Only applicable to JPEG images

#### `extract_entropy(img_array)`
- Calculates entropy for each RGB channel
- Uses histogram analysis with 256 bins

#### `extract_blur_sharpness(img_array)`
- Uses Laplacian variance to measure image sharpness
- Higher values indicate sharper images

#### `extract_hu_moments(img_array)`
- Extracts 7 Hu moments for shape analysis
- Scale, rotation, and translation invariant

#### `extract_dct_features(img_array)`
- Applies DCT to grayscale image
- Extracts statistical measures of DCT coefficients

#### `extract_noise_features(img_array)`
- Estimates noise by subtracting Gaussian-blurred image
- Provides noise statistics

#### `extract_edge_density(img_array)`
- Uses Canny edge detection
- Calculates proportion of edge pixels

#### `extract_image_stats(img)`
- Basic image dimensions
- Width and height in pixels

## 📈 Feature Statistics Summary

### Total Features: 25
- **Basic Info**: 7 features
- **EXIF Metadata**: 4 features  
- **Quantization Table**: 4 features
- **Color Entropy**: 3 features
- **Image Quality**: 2 features
- **Hu Moments**: 7 features
- **DCT Features**: 4 features
- **Noise Analysis**: 2 features

### Data Processing Pipeline
1. **Image Loading**: Convert to RGB format
2. **Format Detection**: Determine image format
3. **EXIF Extraction**: Parse metadata (JPEG/TIFF only)
4. **Quantization Analysis**: Extract JPEG table stats
5. **Color Analysis**: Calculate RGB entropy
6. **Quality Assessment**: Measure blur and edge density
7. **Shape Analysis**: Extract Hu moments
8. **Frequency Analysis**: Apply DCT and extract features
9. **Noise Analysis**: Estimate and characterize noise

## 🎯 Use Cases

### Authenticity Detection
- **EXIF Analysis**: Detect editing software signatures
- **Quantization Tables**: Identify JPEG compression artifacts
- **Noise Patterns**: Distinguish between natural and artificial noise
- **DCT Features**: Detect compression and manipulation artifacts

### Image Classification
- **Hu Moments**: Shape-based classification
- **Color Entropy**: Texture and color distribution analysis
- **Quality Metrics**: Assess image processing history

### Forensic Analysis
- **Metadata Extraction**: Camera and software identification
- **Compression Analysis**: JPEG quality and processing history
- **Noise Characterization**: Detect artificial modifications

## 📋 Output Format

The `process_images()` function returns a list of dictionaries, where each dictionary contains:
- All extracted features for a single image
- Consistent feature names across all images
- `None` values for features not applicable to specific formats
- Error handling for corrupted or unsupported images

## 🔍 Feature Importance

### High-Value Features for Authenticity Detection
1. **EXIF Software** - Direct indicator of editing software
2. **Quantization Table Statistics** - JPEG compression fingerprints
3. **Noise Analysis** - Natural vs. artificial noise patterns
4. **DCT Features** - Frequency domain manipulation detection
5. **Blur Metric** - Sharpness consistency analysis

### Supporting Features
- **Color Entropy** - Texture and color distribution
- **Hu Moments** - Shape and structural analysis
- **Edge Density** - Edge preservation analysis
- **Basic Metadata** - Format and dimension consistency
