"""
Binary Mask Editor - Minimalist PySide6 GUI for editing segmentation masks.
"""

import dataclasses
import sys
import csv
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QFileDialog,
    QColorDialog, QSizePolicy, QFrame, QSlider
)
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, Signal
from PySide6.QtGui import (
    QPainter, QImage, QPixmap, QColor, QPen, QBrush, QMouseEvent, QPaintEvent
)
from dataclasses import dataclass

from spectral_segmenation import segment, save_binary_image

MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 150
PIXEL_SIZE_TO_MILIMETERS = 10e-6

@dataclass
class SegmentationResult():
    image_path:Path
    area:float
    lx: float
    ly: float
    pixel_size: float

    def get(self)->tuple[str, float, float, float]:
        area_m = self.area * self.pixel_size
        lx_m = self.lx * self.pixel_size
        ly_m = self.ly * self.pixel_size
        return (str(self.image_path), area_m, lx_m, ly_m)

class MaskCanvas(QWidget):
    """Canvas widget for displaying image with mask overlay and editing."""
    
    brush_size_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.mask_color = QColor(255, 0, 0, 128)  # Semi-transparent red
        self.brush_size = 15
        self.drawing = False
        self.erasing = False
        self.panning = False
        self.last_point = QPointF()
        self.last_image_point: tuple[int, int] | None = None
        
        # Zoom and pan
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._fit_scale = 1.0  # Scale to fit image in widget
        
        # Cached pixmaps for performance
        self._image_pixmap: QPixmap | None = None
        self._mask_pixmap: QPixmap | None = None
        self._mask_dirty = True
        
        # Precompute brush kernel for speed
        self._brush_kernel: np.ndarray | None = None
        self._brush_kernel_size = -1
        
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_image(self, image: np.ndarray):
        """Set the background image (numpy array, RGB or grayscale)."""
        self.image = image
        self._image_pixmap = QPixmap.fromImage(self._numpy_to_qimage(image))
        self._update_fit_scale()
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self.update()
    
    def set_mask(self, mask: np.ndarray):
        """Set the binary mask (numpy array, same dimensions as image)."""
        self.mask = mask.astype(np.uint8)
        self._mask_dirty = True
        self.update()
    
    def get_mask(self) -> np.ndarray | None:
        """Return the current mask."""
        return self.mask
    
    def set_mask_color(self, color: QColor):
        """Set the mask overlay color."""
        self.mask_color = color
        self._mask_dirty = True
        self.update()
    
    def set_brush_size(self, size: int):
        """Set the brush size for drawing."""
        self.brush_size = size
        self._update_brush_kernel()
    
    def _update_brush_kernel(self):
        """Precompute circular brush kernel."""
        if self._brush_kernel_size == self.brush_size:
            return
        r = self.brush_size
        y, x = np.ogrid[-r:r+1, -r:r+1]
        self._brush_kernel = (x*x + y*y <= r*r)
        self._brush_kernel_size = self.brush_size
    
    def _update_fit_scale(self):
        """Calculate scale to fit image in widget."""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        scale_x = self.width() / w
        scale_y = self.height() / h
        self._fit_scale = min(scale_x, scale_y)
    
    def _get_transform_params(self) -> tuple[float, QPointF]:
        """Get current scale and offset for rendering."""
        if self.image is None:
            return 1.0, QPointF(0, 0)
        
        h, w = self.image.shape[:2]
        scale = self._fit_scale * self._zoom
        
        # Centered + pan
        scaled_w = w * scale
        scaled_h = h * scale
        offset = QPointF(
            (self.width() - scaled_w) / 2 + self._pan_offset.x(),
            (self.height() - scaled_h) / 2 + self._pan_offset.y()
        )
        return scale, offset
    
    def _widget_to_image(self, pos: QPointF) -> tuple[int, int] | None:
        """Convert widget coordinates to image coordinates."""
        if self.image is None:
            return None
        
        scale, offset = self._get_transform_params()
        x = int((pos.x() - offset.x()) / scale)
        y = int((pos.y() - offset.y()) / scale)
        h, w = self.image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None
    
    def _numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        """Convert numpy array to QImage."""
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            h, w = arr.shape
            return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        elif arr.ndim == 3:
            h, w, c = arr.shape
            if c == 3:
                return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
            elif c == 4:
                return QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    
    def _update_mask_pixmap(self):
        """Rebuild the mask overlay pixmap."""
        if self.mask is None:
            self._mask_pixmap = None
            return
        
        mask_rgba = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        mask_rgba[self.mask > 0] = [
            self.mask_color.red(),
            self.mask_color.green(),
            self.mask_color.blue(),
            self.mask_color.alpha()
        ]
        self._mask_pixmap = QPixmap.fromImage(self._numpy_to_qimage(mask_rgba))
        self._mask_dirty = False
    
    def paintEvent(self, event: QPaintEvent):
        """Paint the image and mask overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self.image is None or self._image_pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        
        scale, offset = self._get_transform_params()
        target_rect = QRect(
            int(offset.x()), int(offset.y()),
            int(self._image_pixmap.width() * scale),
            int(self._image_pixmap.height() * scale)
        )
        
        # Draw image
        painter.drawPixmap(target_rect, self._image_pixmap)
        
        # Draw mask overlay
        if self._mask_dirty:
            self._update_mask_pixmap()
        
        if self._mask_pixmap is not None:
            painter.drawPixmap(target_rect, self._mask_pixmap)
    
    def _draw_on_mask(self, pos: QPointF, erase: bool = False):
        """Draw or erase on the mask at the given position."""
        coords = self._widget_to_image(pos)
        if coords is None or self.mask is None:
            return
        
        x, y = coords
        h, w = self.mask.shape
        r = self.brush_size
        
        # Ensure brush kernel is ready
        self._update_brush_kernel()
        
        # Calculate bounds
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        
        # Kernel slice (handle edge cases)
        kx0, kx1 = r - (x - x0), r + (x1 - x)
        ky0, ky1 = r - (y - y0), r + (y1 - y)
        
        kernel_slice = self._brush_kernel[ky0:ky1, kx0:kx1]
        
        # Apply to mask
        if erase:
            self.mask[y0:y1, x0:x1][kernel_slice] = 0
        else:
            self.mask[y0:y1, x0:x1][kernel_slice] = 1
        
        # Interpolate between last point and current for smooth strokes
        if self.last_image_point is not None:
            lx, ly = self.last_image_point
            dist = max(abs(x - lx), abs(y - ly))
            if dist > 1:
                steps = max(2, dist // max(1, r // 2))
                for i in range(1, steps):
                    t = i / steps
                    ix = int(lx + (x - lx) * t)
                    iy = int(ly + (y - ly) * t)
                    
                    ix0, ix1 = max(0, ix - r), min(w, ix + r + 1)
                    iy0, iy1 = max(0, iy - r), min(h, iy + r + 1)
                    ikx0, ikx1 = r - (ix - ix0), r + (ix1 - ix)
                    iky0, iky1 = r - (iy - iy0), r + (iy1 - iy)
                    
                    k_slice = self._brush_kernel[iky0:iky1, ikx0:ikx1]
                    if erase:
                        self.mask[iy0:iy1, ix0:ix1][k_slice] = 0
                    else:
                        self.mask[iy0:iy1, ix0:ix1][k_slice] = 1
        
        self.last_image_point = (x, y)
        self._mask_dirty = True
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for drawing or panning."""
        if self.mask is None and event.button() != Qt.MiddleButton:
            return
        
        pos = event.position()
        
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.erasing = False
            self.last_image_point = None
            self._draw_on_mask(pos, erase=False)
        elif event.button() == Qt.RightButton:
            self.drawing = False
            self.erasing = True
            self.last_image_point = None
            self._draw_on_mask(pos, erase=True)
        elif event.button() == Qt.MiddleButton:
            self.panning = True
            self.setCursor(Qt.ClosedHandCursor)
        
        self.last_point = pos
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for continuous drawing or panning."""
        pos = event.position()
        
        if self.panning:
            delta = pos - self.last_point
            self._pan_offset += QPointF(delta.x(), delta.y())
            self.update()
        elif self.drawing:
            self._draw_on_mask(pos, erase=False)
        elif self.erasing:
            self._draw_on_mask(pos, erase=True)
        
        self.last_point = pos
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            self.drawing = False
            self.erasing = False
            self.last_image_point = None
    
    def wheelEvent(self, event):
        """Handle scroll wheel for zoom."""
        if self.image is None:
            return
        
        # Zoom centered on mouse position
        old_pos = self._widget_to_image(event.position())
        
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = self._zoom * factor
        new_zoom = max(0.1, min(20.0, new_zoom))  # Clamp zoom
        
        if new_zoom != self._zoom:
            # Adjust pan to zoom towards mouse
            scale_old, offset_old = self._get_transform_params()
            self._zoom = new_zoom
            scale_new, _ = self._get_transform_params()
            
            if old_pos:
                # Keep the point under mouse stationary
                mx, my = event.position().x(), event.position().y()
                ix, iy = old_pos
                new_offset_x = mx - ix * scale_new - (self.width() - self.image.shape[1] * scale_new) / 2
                new_offset_y = my - iy * scale_new - (self.height() - self.image.shape[0] * scale_new) / 2
                self._pan_offset = QPointF(new_offset_x, new_offset_y)
            
            self.update()
    
    def reset_view(self):
        """Reset zoom and pan to default."""
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize to update scaling."""
        self._update_fit_scale()
        super().resizeEvent(event)


class MaskEditor(QMainWindow):
    """Main window for the mask editor application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binary Mask Editor")
        self.setMinimumSize(800, 600)
        
        # Data storage
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.seg_results: list[np.ndarray] = []
        self.filenames: list[str] = []
        self.current_index = 0
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # === Header with parameters ===
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        
        # Pixel Size
        pixel_size_label = QLabel("Pixel Size:")
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0, 1000)
        self.pixel_size_spin.setValue(8.02)
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setFixedWidth(100)
        
        # Binary Threshold
        binary_threshold_label = QLabel("Binary Threshold:")
        self.binary_threshold_spin = QSpinBox()
        self.binary_threshold_spin.setRange(0, 1000)
        self.binary_threshold_spin.setValue(100)
        self.binary_threshold_spin.setFixedWidth(100)
        
        # Low-pass Radius
        lo_pass_radius_label = QLabel("Lo-pass Radius:")
        self.lo_pass_radius_spin = QDoubleSpinBox()
        self.lo_pass_radius_spin.setRange(0, 1000)
        self.lo_pass_radius_spin.setValue(17.0)
        self.lo_pass_radius_spin.setDecimals(2)
        self.lo_pass_radius_spin.setFixedWidth(100)
        
        header_layout.addWidget(pixel_size_label)
        header_layout.addWidget(self.pixel_size_spin)
        header_layout.addSpacing(20)
        header_layout.addWidget(binary_threshold_label)
        header_layout.addWidget(self.binary_threshold_spin)
        header_layout.addSpacing(20)
        header_layout.addWidget(lo_pass_radius_label)
        header_layout.addWidget(self.lo_pass_radius_spin)
        header_layout.addSpacing(20)
        
        # Run segmentation button
        self.run_seg_btn = QPushButton("Run Segmentation")
        self.run_seg_btn.clicked.connect(self._run_segmentation)
        self.run_seg_btn.setObjectName("runSegBtn")
        header_layout.addWidget(self.run_seg_btn)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # === Canvas ===
        self.canvas = MaskCanvas()
        layout.addWidget(self.canvas, stretch=1)
        
        # === Brush size slider ===
        brush_frame = QFrame()
        brush_frame.setObjectName("brushFrame")
        brush_layout = QHBoxLayout(brush_frame)
        brush_layout.setContentsMargins(10, 6, 10, 6)
        
        brush_label = QLabel("Brush Size:")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.brush_slider.setValue(15)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        
        self.brush_value_label = QLabel("15")
        self.brush_value_label.setFixedWidth(30)
        self.brush_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.canvas.reset_view)
        self.reset_view_btn.setFixedWidth(90)
        
        brush_layout.addWidget(brush_label)
        brush_layout.addWidget(self.brush_slider, stretch=1)
        brush_layout.addWidget(self.brush_value_label)
        brush_layout.addSpacing(20)
        brush_layout.addWidget(self.reset_view_btn)
        
        layout.addWidget(brush_frame)
        
        # === Controls ===
        controls = QFrame()
        controls.setObjectName("controls")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        
        # Load directory button
        self.load_btn = QPushButton("Load Directory")
        self.load_btn.clicked.connect(self._load_directory)
        
        # Navigation
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._prev_image)
        self.prev_btn.setEnabled(False)
        
        self.image_label = QLabel("No images")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumWidth(150)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._next_image)
        self.next_btn.setEnabled(False)
        
        # Color picker
        self.color_btn = QPushButton("Mask Color")
        self.color_btn.clicked.connect(self._pick_color)
        self._update_color_button()
        
        # Save button
        self.save_btn = QPushButton("Save All")
        self.save_btn.clicked.connect(self._save_masks)
        self.save_btn.setEnabled(False)
        self.save_btn.setObjectName("saveBtn")
        
        controls_layout.addWidget(self.load_btn)
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.image_label)
        controls_layout.addWidget(self.next_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.color_btn)
        controls_layout.addSpacing(15)
        controls_layout.addWidget(self.save_btn)
        
        layout.addWidget(controls)
        
        # === Status bar hint ===
        self.statusBar().showMessage("Left-click: draw | Right-click: erase | Scroll: zoom | Middle-click: pan")
    
    def _on_brush_size_changed(self, value: int):
        """Handle brush size slider change."""
        self.canvas.set_brush_size(value)
        self.brush_value_label.setText(str(value))
    
    def _run_segmentation(self):
        """Run automatic segmentation with current parameters. (Dummy implementation)"""
        pixel_size = self.pixel_size_spin.value() * PIXEL_SIZE_TO_MILIMETERS # input pixel size to MM scale (e.g. from micrometers)
        binary_threshold = self.binary_threshold_spin.value()
        lo_pass_radius = self.lo_pass_radius_spin.value()
        
        # TODO: Implement actual segmentation logic
        self._perform_segmentation(pixel_size, binary_threshold, lo_pass_radius)
    
    def _perform_segmentation(self, pixel_size: float, binary_threshold: int, lo_pass_radius: float):
        """
        Perform automatic segmentation on the current image.
        
        Args:
            pixel_size: Pixel size parameter
            binary_threshold: Binary threshold parameter
            lo_pass_radius: Low-pass filter radius parameter
        
        Override this method or connect to run_segmentation for custom logic.
        """
        # Dummy implementation - fill with placeholder message
        if not self.images:
            self.statusBar().showMessage("No image loaded for segmentation")
            return
        
        self.statusBar().showMessage(f"Running segmentation with pixel_size={pixel_size}, binary_threshold={binary_threshold}, lo_pass_radius={lo_pass_radius}...")
        
        # # Placeholder: create a simple threshold-based mask as demo
        # # Replace this with actual segmentation algorithm
        # image = self.images[self.current_index]
        # if image.ndim == 3:
        #     gray = np.mean(image, axis=2)
        # else:
        #     gray = image.astype(float)
        
        # # Dummy segmentation: simple threshold (replace with real algorithm)
        # threshold = binary_threshold if binary_threshold > 0 else 128
        # mask = (gray > threshold).astype(np.uint8)

        img = self.images[self.current_index]
        mask, area, lx, ly = segment(img, binary_threshold, lo_pass_radius)

        self.masks[self.current_index] = mask
        self.seg_results[self.current_index] = SegmentationResult(
            self.filenames[self.current_index],
            area,
            lx,
            ly,
            pixel_size
        )

        self.canvas.set_mask(mask)
        self.statusBar().showMessage(f"Segmentation complete (pixel_size={pixel_size}, binary_threshold={binary_threshold}, lo_pass_radius={lo_pass_radius})")



    def _apply_style(self):
        """Apply minimal dark stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QFrame#header, QFrame#controls, QFrame#brushFrame {
                background-color: #2d2d2d;
                border-radius: 6px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #5d5d5d;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
            QPushButton#saveBtn {
                background-color: #2e7d32;
            }
            QPushButton#saveBtn:hover {
                background-color: #388e3c;
            }
            QPushButton#saveBtn:disabled {
                background-color: #1b4d1e;
                color: #666666;
            }
            QPushButton#runSegBtn {
                background-color: #1565c0;
            }
            QPushButton#runSegBtn:hover {
                background-color: #1976d2;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #4d4d4d;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3d3d3d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6d6d6d;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #8d8d8d;
            }
            QSlider::sub-page:horizontal {
                background: #5a5a5a;
                border-radius: 3px;
            }
            QStatusBar {
                color: #888888;
                font-size: 12px;
            }
        """)
    
    def _update_color_button(self):
        """Update the color button appearance."""
        color = self.canvas.mask_color
        self.color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba({color.red()}, {color.green()}, {color.blue()}, 200);
                color: {'#000' if color.lightness() > 128 else '#fff'};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: rgba({color.red()}, {color.green()}, {color.blue()}, 230);
            }}
        """)
    
    def _load_directory(self):
        """Load all images from a directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Directory", "",
            QFileDialog.ShowDirsOnly
        )
        if not dir_path:
            return
        
        self._load_images_from_directory(Path(dir_path))
    
    def _load_images_from_directory(self, dir_path: Path):
        """Load images from the given directory."""
        import cv2
        
        self.images.clear()
        self.masks.clear()
        self.filenames.clear()
        
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.jp2'}
        files = sorted([
            f for f in dir_path.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            
            # Convert BGR to RGB if needed
            if img.ndim == 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to 8-bit if needed
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            self.images.append(img)
            self.masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
            self.seg_results.append(None)
            self.filenames.append(f.stem)
        
        if self.images:
            self.current_index = 0
            self._update_display()
            self._update_navigation()
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded {len(self.images)} images from {dir_path.name}")
        else:
            self.statusBar().showMessage("No valid images found in directory")
    
    def _update_display(self):
        """Update the canvas with current image and mask."""
        if not self.images:
            return
        
        # Store current mask before switching
        if hasattr(self, '_last_index') and self._last_index != self.current_index:
            if self.canvas.get_mask() is not None:
                self.masks[self._last_index] = self.canvas.get_mask().copy()
        
        self.canvas.set_image(self.images[self.current_index])
        self.canvas.set_mask(self.masks[self.current_index])
        self.image_label.setText(f"{self.filenames[self.current_index]} ({self.current_index + 1}/{len(self.images)})")
        self._last_index = self.current_index
    
    def _update_navigation(self):
        """Update navigation button states."""
        has_images = len(self.images) > 0
        self.prev_btn.setEnabled(has_images and self.current_index > 0)
        self.next_btn.setEnabled(has_images and self.current_index < len(self.images) - 1)
    
    def _prev_image(self):
        """Go to previous image."""
        if self.current_index > 0:
            self._store_current_mask()
            self.current_index -= 1
            self._update_display()
            self._update_navigation()
    
    def _next_image(self):
        """Go to next image."""
        if self.current_index < len(self.images) - 1:
            self._store_current_mask()
            self.current_index += 1
            self._update_display()
            self._update_navigation()
    
    def _store_current_mask(self):
        """Store the current mask back to the list."""
        if self.images and self.canvas.get_mask() is not None:
            self.masks[self.current_index] = self.canvas.get_mask().copy()
    
    def _pick_color(self):
        """Open color picker for mask color."""
        color = QColorDialog.getColor(
            self.canvas.mask_color,
            self,
            "Select Mask Color",
            QColorDialog.ShowAlphaChannel
        )
        if color.isValid():
            # Ensure some transparency
            if color.alpha() == 255:
                color.setAlpha(128)
            self.canvas.set_mask_color(color)
            self._update_color_button()
    
    def _save_masks(self):
        """Save all masks to selected directory."""
        if not self.masks:
            return
        
        # Store current mask first
        self._store_current_mask()
        
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "",
            QFileDialog.ShowDirsOnly
        )
        if not out_dir:
            return
        
        import cv2
        out_path = Path(out_dir)
        
        for i, (mask, name) in enumerate(zip(self.masks, self.filenames)):
            mask_file = out_path / f"{name}_mask.png"
            # Save as binary mask (0 or 255)
            cv2.imwrite(str(mask_file), mask * 255)

            save_binary_image(
                mask,
                out_path / f"{name}_segmentation.png",
                description=name,
                units_per_pixel=self.pixel_size_spin.value() * PIXEL_SIZE_TO_MILIMETERS
            )

        out_file = out_path / "result.csv"
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["File Name", "Area", "lx", "ly"])
            for idx, seg_res in enumerate(self.seg_results):
                if seg_res is None:
                    writer.writerow([self.filenames[idx], "", "", ""])
                else:
                    writer.writerow(seg_res.get())
        
        
        self.statusBar().showMessage(f"Saved {len(self.masks)} masks to {out_path.name}")

    
    def load_from_arrays(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None, names: list[str] | None = None):
        """
        Load images and masks from numpy arrays programmatically.
        
        Args:
            images: List of numpy arrays (RGB or grayscale)
            masks: Optional list of binary masks (same dimensions as images)
            names: Optional list of names for each image
        """
        self.images = images
        self.masks = masks if masks else [np.zeros(img.shape[:2], dtype=np.uint8) for img in images]
        self.filenames = names if names else [f"image_{i}" for i in range(len(images))]
        
        if self.images:
            self.current_index = 0
            self._update_display()
            self._update_navigation()
            self.save_btn.setEnabled(True)
    
    def get_parameters(self) -> tuple[float, int, float]:
        """Get the current parameter values (pixel_size, binary_threshold, lo_pass_radius)."""
        return self.pixel_size_spin.value(), self.binary_threshold_spin.value(), self.lo_pass_radius_spin.value()
    
    def set_parameters(self, pixel_size: float, binary_threshold: int, lo_pass_radius: float):
        """Set the parameter values."""
        self.pixel_size_spin.setValue(pixel_size)
        self.binary_threshold_spin.setValue(binary_threshold)
        self.lo_pass_radius_spin.setValue(lo_pass_radius)


def main():
    """Run the mask editor application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    editor = MaskEditor()
    editor.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
