from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, 0.0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1000:
        return image, 0.0
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 2:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation, (w, h))
        return rotated, angle
    return image, 0.0


def correct_perspective(image: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1000:
        return image, {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        return image, {"x": 0, "y": 0, "w": image.shape[1], "h": image.shape[0]}
    rect = order_points(approx.reshape(4, 2).astype("float32"))
    (w, h) = image.shape[1], image.shape[0]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    corrected = cv2.warpPerspective(image, matrix, (w, h))
    x, y, w_box, h_box = cv2.boundingRect(contour)
    return corrected, {"x": x, "y": y, "w": w_box, "h": h_box}


def preprocess_image(image: np.ndarray, upscale: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    if upscale > 1.0:
        image = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.bilateralFilter(corrected, 9, 75, 75)
    deskewed, _ = deskew_image(denoised)
    perspective_corrected, board_box = correct_perspective(deskewed)
    gray = cv2.cvtColor(perspective_corrected, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return perspective_corrected, gray, board_box
