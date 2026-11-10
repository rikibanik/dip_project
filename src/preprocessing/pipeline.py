from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class ProcessedVariant:
    name: str
    image: np.ndarray
    parameters: dict

def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image is blank")
    return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def resize_image(image: np.ndarray, width: int = 1800) -> np.ndarray:
    if width < 1: raise ValueError("width must be positive")
    h, w = image.shape[:2]
    if w <= width: return image.copy()
    return cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)

def grayscale(image: np.ndarray) -> np.ndarray: return _gray(image)
def histogram_equalize(image: np.ndarray) -> np.ndarray: return cv2.equalizeHist(_gray(image))
def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if clip_limit <= 0 or tile_size < 1: raise ValueError("CLAHE parameters must be positive")
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(_gray(image))
def gaussian_denoise(image: np.ndarray, kernel: int = 3) -> np.ndarray:
    if kernel < 1 or kernel % 2 == 0: raise ValueError("kernel must be positive and odd")
    return cv2.GaussianBlur(_gray(image), (kernel, kernel), 0)
def median_denoise(image: np.ndarray, kernel: int = 3) -> np.ndarray:
    if kernel < 3 or kernel % 2 == 0: raise ValueError("median kernel must be odd and at least 3")
    return cv2.medianBlur(_gray(image), kernel)
def global_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    if not 0 <= threshold <= 255: raise ValueError("threshold must be in [0,255]")
    return cv2.threshold(_gray(image), threshold, 255, cv2.THRESH_BINARY)[1]
def otsu_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.threshold(_gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def adaptive_threshold(image: np.ndarray, block_size: int = 31, c: int = 10) -> np.ndarray:
    if block_size < 3 or block_size % 2 == 0: raise ValueError("block_size must be odd and >= 3")
    return cv2.adaptiveThreshold(_gray(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

def morphology(image: np.ndarray, operation: str = "open", kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    if kernel_size < 1 or iterations < 1: raise ValueError("kernel_size and iterations must be positive")
    ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    if operation not in ops: raise ValueError(f"Unknown morphology operation: {operation}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(_gray(image), ops[operation], kernel, iterations=iterations)

def detect_document_region(image: np.ndarray) -> tuple[np.ndarray, list[tuple[int,int,int,int]]]:
    gray = _gray(image); edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > image.shape[0]*image.shape[1]*0.005]
    boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
    return edges, boxes

def connected_components(image: np.ndarray) -> list[dict]:
    binary = otsu_threshold(image)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(255-binary, 8)
    return [{"x":int(stats[i,0]),"y":int(stats[i,1]),"width":int(stats[i,2]),"height":int(stats[i,3]),"area":int(stats[i,4]),"centroid":centroids[i].tolist()}
            for i in range(1,n) if stats[i,cv2.CC_STAT_AREA] > 2]

def segment_text_lines(image: np.ndarray, min_height: int = 3) -> list[tuple[int,int]]:
    binary = otsu_threshold(image)
    projection = np.count_nonzero(binary == 0, axis=1)
    occupied = projection > max(1, int(binary.shape[1]*0.002))
    spans=[]; start=None
    for i, value in enumerate(occupied.tolist()+[False]):
        if value and start is None: start=i
        elif not value and start is not None:
            if i-start >= min_height: spans.append((start,i))
            start=None
    return spans

def estimate_skew(image: np.ndarray) -> float:
    binary = otsu_threshold(image)
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 20: return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    return float(-(90 + angle) if angle < -45 else -angle)

def deskew(image: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray,float]:
    angle=estimate_skew(image)
    if abs(angle) > max_angle: return _gray(image), angle
    h,w=image.shape[:2]; matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(_gray(image),matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE), angle

def build_variants(image: np.ndarray) -> dict[str, ProcessedVariant]:
    """Return independent OCR views, including the deskewed grayscale image."""
    g=grayscale(image); den=median_denoise(g)
    deskewed,skew_angle=deskew(g)
    items={"original":(image,{"operation":"identity"}),"grayscale":(g,{"operation":"grayscale"}),
           "histogram_equalized":(histogram_equalize(g),{"operation":"histogram_equalize"}),
           "clahe":(clahe(g),{"operation":"clahe","clip_limit":2.0,"tile_size":8}),
           "global_threshold":(global_threshold(g),{"operation":"global_threshold","threshold":127}),
           "otsu_threshold":(otsu_threshold(g),{"operation":"otsu_threshold"}),
           "adaptive_threshold":(adaptive_threshold(g),{"operation":"adaptive_threshold","block_size":31,"c":10}),
           "gaussian_denoised":(gaussian_denoise(g),{"operation":"gaussian_denoise","kernel":3}),
           "denoised":(den,{"operation":"median_denoise","kernel":3}),
           "morphology":(morphology(otsu_threshold(g),"open",2),{"operation":"morphology","mode":"open","kernel_size":2}),
           "deskewed":(deskewed,{"operation":"deskew","estimated_angle_degrees":skew_angle,"max_angle_degrees":15.0,"interpolation":"cubic","border_mode":"replicate"})}
    return {k:ProcessedVariant(k,v,p) for k,(v,p) in items.items()}
