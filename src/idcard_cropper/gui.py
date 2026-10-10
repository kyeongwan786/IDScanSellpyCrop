"""Desktop GUI for selecting, correcting, and printing ID card scans."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QUrl, QSizeF, QSettings, QPointF
from PySide6.QtGui import (QDesktopServices, QImage, QPageLayout, QPageSize, QPainter,
                           QPixmap, QPen, QColor, QPolygonF)
from PySide6.QtPrintSupport import QPrinter, QPrinterInfo
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .engine import CropResult, _load_image, _rect_corners, _rectified_crop, process_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
USER_ROLE = int(Qt.ItemDataRole.UserRole)
SETTINGS_ORGANIZATION = "IDCardCropper"
SETTINGS_APPLICATION = "IDCardCropper"


def _preferred_printer_name() -> str:
    names = QPrinterInfo.availablePrinterNames()
    settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
    saved = str(settings.value("printing/last_printer", ""))
    if saved in names:
        return saved
    selphy = ""
    for name in names:
        info = QPrinterInfo.printerInfo(name)
        identity = f"{name} {info.description()} {info.makeAndModel()}".casefold()
        if "selphy" in identity:
            selphy = name
            break
    if selphy:
        return selphy
    default = QPrinterInfo.defaultPrinterName()
    return default if default in names else (names[0] if names else "")


def _c_size_for_printer(printer: QPrinter) -> QPageSize:
    info = QPrinterInfo(printer)
    for supported_size in info.supportedPageSizes():
        size = supported_size.size(QPageSize.Unit.Millimeter)
        if abs(max(size.width(), size.height()) - 86.0) <= 0.5 and abs(
            min(size.width(), size.height()) - 54.0
        ) <= 0.5:
            return supported_size
    return QPageSize(QSizeF(86.0, 54.0), QPageSize.Unit.Millimeter, "C Size")


def _set_c_landscape_default(printer: QPrinter) -> None:
    printer.setPageSize(_c_size_for_printer(printer))
    printer.setPageOrientation(QPageLayout.Orientation.Landscape)


class Preview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #20242b; color: #d7dbe2; border: 1px solid #444b55; }")
        self._pixmap: QPixmap | None = None
        self._placeholder = title

    def show_image(self, path: Path | None) -> None:
        self._pixmap = QPixmap(str(path)) if path and path.is_file() else None
        self._render()

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap and not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
        else:
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)


class CropEditor(Preview):
    """Interactive rotated-rectangle editor over the source scan."""

    rectangle_changed = Signal(object)
    HANDLE_RADIUS = 12.0

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._source: np.ndarray | None = None
        self.rectangle: tuple[float, float, float, float, float] | None = None
        self._drag: str | None = None
        self._drag_start = QPointF()
        self._drag_rect: tuple[float, float, float, float, float] | None = None

    def load_source(self, path: Path | None) -> None:
        self._source = _load_image(path) if path and path.is_file() else None
        self.show_image(path)
        if self._source is None:
            self.set_rectangle(None)

    def set_rectangle(self, rect) -> None:
        self.rectangle = tuple(float(value) for value in rect) if rect else None
        self.update()

    def _image_view(self) -> tuple[QRectF, float]:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF(), 1.0
        bounds = self.contentsRect()
        size = self._pixmap.size().scaled(bounds.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = bounds.x() + (bounds.width() - size.width()) / 2.0
        y = bounds.y() + (bounds.height() - size.height()) / 2.0
        scale = size.width() / max(1, self._pixmap.width())
        return QRectF(x, y, size.width(), size.height()), scale

    def _to_view(self, point) -> QPointF:
        bounds, scale = self._image_view()
        return QPointF(bounds.x() + float(point[0]) * scale,
                       bounds.y() + float(point[1]) * scale)

    def _to_image(self, point: QPointF) -> np.ndarray:
        bounds, scale = self._image_view()
        return np.array([(point.x() - bounds.x()) / scale,
                         (point.y() - bounds.y()) / scale], dtype=np.float32)

    def _handle_points(self) -> dict[str, QPointF]:
        if self.rectangle is None:
            return {}
        corners = _rect_corners(self.rectangle)
        center = np.array(self.rectangle[:2], dtype=np.float32)
        theta = np.deg2rad(self.rectangle[4])
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        cx, cy, width, height, _ = self.rectangle
        return {
            "left": self._to_view((center - u * width / 2).tolist()),
            "right": self._to_view((center + u * width / 2).tolist()),
            "top": self._to_view((center - v * height / 2).tolist()),
            "bottom": self._to_view((center + v * height / 2).tolist()),
            "rotate": self._to_view((center - v * (height / 2 + 38 / max(self._image_view()[1], 1e-6))).tolist()),
        }

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().paintEvent(event)
        if self.rectangle is None or self._pixmap is None or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(20, 220, 100), 2.5))
        polygon = QPolygonF([self._to_view(point) for point in _rect_corners(self.rectangle)])
        painter.drawPolygon(polygon)
        center = self._to_view(self.rectangle[:2])
        handles = self._handle_points()
        painter.drawLine(polygon[0], handles["rotate"])
        painter.setBrush(QColor(255, 255, 255))
        for name in ("left", "right", "top", "bottom"):
            painter.drawEllipse(handles[name], 6.0, 6.0)
        painter.setBrush(QColor(255, 190, 30))
        painter.drawEllipse(handles["rotate"], 7.0, 7.0)
        painter.setBrush(QColor(20, 220, 100))
        painter.drawEllipse(center, 4.0, 4.0)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if event.button() != Qt.MouseButton.LeftButton or self.rectangle is None:
            return super().mousePressEvent(event)
        point = event.position()
        handles = self._handle_points()
        for name, handle in handles.items():
            if np.hypot(point.x() - handle.x(), point.y() - handle.y()) <= self.HANDLE_RADIUS:
                self._drag = name
                break
        if self._drag is None:
            polygon = QPolygonF([self._to_view(p) for p in _rect_corners(self.rectangle)])
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                self._drag = "move"
        if self._drag:
            self._drag_start = self._to_image(point)
            self._drag_rect = self.rectangle
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if not self._drag or self._drag_rect is None:
            return super().mouseMoveEvent(event)
        start, now = self._drag_start, self._to_image(event.position())
        dx, dy = now - start
        cx, cy, width, height, angle = self._drag_rect
        theta = np.deg2rad(angle)
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        if self._drag == "move":
            cx, cy = cx + float(dx), cy + float(dy)
        elif self._drag == "rotate":
            before = np.arctan2(start[1] - cy, start[0] - cx)
            after = np.arctan2(now[1] - cy, now[0] - cx)
            angle += float(np.degrees(after - before))
        elif self._drag in ("left", "right"):
            delta = float(np.dot(np.array([dx, dy]), u))
            sign = -1.0 if self._drag == "left" else 1.0
            width = max(24.0, width + sign * delta)
            cx += float(u[0] * delta / 2)
            cy += float(u[1] * delta / 2)
        else:
            delta = float(np.dot(np.array([dx, dy]), v))
            sign = -1.0 if self._drag == "top" else 1.0
            height = max(24.0, height + sign * delta)
            cx += float(v[0] * delta / 2)
            cy += float(v[1] * delta / 2)
        self.rectangle = (cx, cy, width, height, angle)
        self.update()
        self.rectangle_changed.emit(self.rectangle)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        self._drag = None
        self._drag_rect = None
        self.setCursor(Qt.CursorShape.ArrowCursor)


class ProcessWorker(QThread):
    item_done = Signal(str, object)

    def __init__(self, paths: list[Path], output_dir: Path, *, crop_mode: str = "rectangle",
                 manual_rect=None) -> None:
        super().__init__()
        self.paths = paths
        self.output_dir = output_dir
        self.crop_mode = crop_mode
        self.manual_rect = manual_rect

    def run(self) -> None:
        for path in self.paths:
            result = process_image(path, self.output_dir, crop_mode=self.crop_mode,
                                   manual_rect=self.manual_rect)
            self.item_done.emit(str(path), result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("신분증 스캔 보정 및 인쇄")
        self.resize(1120, 720)
        self.results: dict[str, CropResult] = {}
        self.worker: ProcessWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        self.add_files_button = QPushButton("이미지 추가")
        self.add_folder_button = QPushButton("폴더 추가")
        self.remove_button = QPushButton("목록에서 제거")
        toolbar.addWidget(self.add_files_button)
        toolbar.addWidget(self.add_folder_button)
        toolbar.addWidget(self.remove_button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("결과 폴더"))
        self.output_edit = QLineEdit(str(Path.cwd() / "output"))
        self.output_browse_button = QPushButton("찾아보기")
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(self.output_browse_button)
        layout.addLayout(output_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(250)
        splitter.addWidget(self.file_list)

        previews = QWidget()
        preview_layout = QHBoxLayout(previews)
        self.source_preview = CropEditor("원본 미리보기")
        self.result_preview = Preview("보정 결과 미리보기")
        preview_layout.addWidget(self._preview_column("원본", self.source_preview), 1)
        preview_layout.addWidget(self._preview_column("보정 결과", self.result_preview), 1)
        splitter.addWidget(previews)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.crop_help = QLabel(
            "검출된 초록 사각형 안쪽을 드래그해 이동하고, 변 중간의 흰색 핸들로 크기를 조절하세요. "
            "위쪽 주황색 핸들은 전체 회전입니다. 조정 후 ‘수동 사각형 적용’을 누르면 다시 크롭합니다."
        )
        self.crop_help.setWordWrap(True)
        layout.addWidget(self.crop_help)

        actions = QHBoxLayout()
        self.process_selected_button = QPushButton("선택 항목 보정")
        self.process_all_button = QPushButton("전체 보정")
        self.manual_apply_button = QPushButton("수동 사각형 적용")
        self.perspective_mode = QCheckBox("고급 원근 보정")
        self.print_button = QPushButton("보정 결과 인쇄")
        self.open_diagnostics_button = QPushButton("단계별 진단 열기")
        self.open_output_button = QPushButton("결과 폴더 열기")
        actions.addWidget(self.process_selected_button)
        actions.addWidget(self.process_all_button)
        actions.addWidget(self.manual_apply_button)
        actions.addWidget(self.perspective_mode)
        actions.addStretch(1)
        actions.addWidget(self.print_button)
        actions.addWidget(self.open_diagnostics_button)
        actions.addWidget(self.open_output_button)
        layout.addLayout(actions)

        self.status = QLabel("이미지를 추가한 다음 보정을 실행하세요. 원본 파일은 변경하지 않습니다.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.add_files_button.clicked.connect(self.add_files)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.output_browse_button.clicked.connect(self.choose_output)
        self.process_selected_button.clicked.connect(self.process_selected)
        self.process_all_button.clicked.connect(self.process_all)
        self.manual_apply_button.clicked.connect(self.apply_manual_rectangle)
        self.source_preview.rectangle_changed.connect(self.update_live_crop_preview)
        self.file_list.currentItemChanged.connect(self.selection_changed)
        self.print_button.clicked.connect(self.print_selected)
        self.open_diagnostics_button.clicked.connect(self.open_diagnostics)
        self.open_output_button.clicked.connect(self.open_output)
        self.print_button.setEnabled(False)
        self.open_diagnostics_button.setEnabled(False)
        self.manual_apply_button.setEnabled(False)

    @staticmethod
    def _preview_column(title: str, preview: Preview) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(preview, 1)
        return column

    def add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "신분증 이미지 선택", str(Path.home()),
            "이미지 (*.jpg *.jpeg *.png)",
        )
        self.add_paths([Path(path) for path in files])

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택", str(Path.home()))
        if folder:
            self.add_paths(sorted(path for path in Path(folder).iterdir()
                                  if path.suffix.lower() in IMAGE_SUFFIXES))

    def add_paths(self, paths: list[Path]) -> None:
        known = {self.file_list.item(i).data(USER_ROLE) for i in range(self.file_list.count())}
        added = 0
        for path in paths:
            resolved = str(path.resolve())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and resolved not in known:
                item = QListWidgetItem(f"대기 중 · {path.name}")
                item.setData(USER_ROLE, resolved)
                self.file_list.addItem(item)
                known.add(resolved)
                added += 1
        if added:
            self.file_list.setCurrentRow(self.file_list.count() - added)
            self.status.setText(f"이미지 {added}개를 추가했습니다.")

    def remove_selected(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        row = self.file_list.currentRow()
        if row >= 0:
            item = self.file_list.takeItem(row)
            self.results.pop(item.data(USER_ROLE), None)
            self.selection_changed(self.file_list.currentItem(), None)

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "결과 폴더 선택", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def selected_path(self) -> Path | None:
        item = self.file_list.currentItem()
        return Path(item.data(USER_ROLE)) if item else None

    def process_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.start_processing([path])

    def process_all(self) -> None:
        paths = [Path(self.file_list.item(i).data(USER_ROLE))
                 for i in range(self.file_list.count())]
        if paths:
            self.start_processing(paths)

    def start_processing(self, paths: list[Path], *, manual_rect=None) -> None:
        output_dir = Path(self.output_edit.text()).expanduser()
        self.set_busy(True)
        self.status.setText(f"{len(paths)}개 이미지를 처리하고 있습니다…")
        mode = "perspective" if self.perspective_mode.isChecked() else "rectangle"
        if len(paths) != 1 or self.perspective_mode.isChecked():
            manual_rect = None
        self.worker = ProcessWorker(paths, output_dir, crop_mode=mode, manual_rect=manual_rect)
        self.worker.item_done.connect(self.processing_done)
        self.worker.finished.connect(lambda: self.set_busy(False))
        self.worker.start()

    def processing_done(self, raw_path: str, result: CropResult) -> None:
        self.results[raw_path] = result
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(USER_ROLE) == raw_path:
                item.setText(f"{self._status_text(result.status)} · {Path(raw_path).name}")
                break
        current = self.selected_path()
        if current and str(current) == raw_path:
            self.display_path(current)
            message = result.message
            if result.quality_warning:
                message += " " + result.quality_warning
            if result.debug_dir:
                message += f" 단계별 캡처 저장: {result.debug_dir}"
            self.status.setText(message)

    @staticmethod
    def _status_text(status: str) -> str:
        return {"complete": "완료", "manual_review": "수동 확인", "failed": "실패"}.get(status, status)

    def selection_changed(self, current: QListWidgetItem | None, _previous) -> None:
        path = Path(current.data(USER_ROLE)) if current else None
        if path:
            self.display_path(path)
        else:
            self.source_preview.load_source(None)
            self.result_preview.show_image(None)
            self.print_button.setEnabled(False)
            self.open_diagnostics_button.setEnabled(False)

    def display_path(self, path: Path) -> None:
        self.source_preview.load_source(path)
        result = self.results.get(str(path))
        if result:
            self.result_preview.show_image(result.output_path or result.preview_path)
            self.source_preview.set_rectangle(result.rectangle)
            self.print_button.setEnabled(bool(result.output_path and result.output_path.is_file()))
            self.open_diagnostics_button.setEnabled(bool(result.debug_dir and result.debug_dir.is_dir()))
            self.manual_apply_button.setEnabled(result.rectangle is not None)
        else:
            self.result_preview.show_image(None)
            self.source_preview.set_rectangle(None)
            self.print_button.setEnabled(False)
            self.open_diagnostics_button.setEnabled(False)
            self.manual_apply_button.setEnabled(False)

    def apply_manual_rectangle(self) -> None:
        path = self.selected_path()
        rect = self.source_preview.rectangle
        if not path or rect is None:
            QMessageBox.information(self, "사각형 없음", "먼저 이미지를 자동 보정해 검출 사각형을 만든 뒤 조정하세요.")
            return
        self.perspective_mode.setChecked(False)
        self.start_processing([path], manual_rect=rect)

    def update_live_crop_preview(self, rect) -> None:
        if self.source_preview._source is None or rect is None:
            return
        try:
            card = _rectified_crop(self.source_preview._source, tuple(rect))
            card = cv2.resize(card, (957, 602), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            preview = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
            self.result_preview.show_pixmap(QPixmap.fromImage(preview.copy()))
        except (cv2.error, ValueError):
            return

    def open_diagnostics(self) -> None:
        path = self.selected_path()
        result = self.results.get(str(path)) if path else None
        if not result or not result.debug_dir or not result.debug_dir.is_dir():
            QMessageBox.information(self, "진단 자료 없음", "먼저 이미지를 보정하세요.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.debug_dir.resolve())))

    def print_selected(self) -> None:
        path = self.selected_path()
        result = self.results.get(str(path)) if path else None
        if not result or not result.output_path or not result.output_path.is_file():
            QMessageBox.warning(self, "인쇄할 결과 없음", "먼저 이미지를 보정하세요.")
            return

        available = QPrinterInfo.availablePrinterNames()
        if not available:
            QMessageBox.critical(
                self, "프린터를 찾을 수 없음",
                "Windows에 설치된 프린터가 없습니다. SELPHY 드라이버를 설치하고 연결 상태를 확인하세요.",
            )
            return

        preferred_name = _preferred_printer_name()
        # Qt's native Windows print dialog reloads the driver's saved DEVMODE
        # and ignores the page layout seeded on QPrinter. It then shows
        # portrait even though the job is changed back to landscape below.
        # Use an app-owned confirmation dialog so the required C/landscape
        # settings are explicit and applied consistently on every print.
        dialog = QDialog(self)
        dialog.setWindowTitle("SELPHY 인쇄 설정")
        form = QFormLayout(dialog)
        printer_combo = QComboBox(dialog)
        printer_combo.addItems(available)
        if preferred_name in available:
            printer_combo.setCurrentText(preferred_name)
        form.addRow("프린터", printer_combo)
        form.addRow("용지", QLabel("C 카드 (54 × 86 mm)"))
        form.addRow("방향", QLabel("가로"))
        form.addRow("인쇄 크기", QLabel("8.1 × 5.1 cm · 300 DPI"))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Print | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_printer = printer_combo.currentText()
        settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        settings.setValue("printing/last_printer", selected_printer)
        settings.sync()

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPrinterName(selected_printer)
        _set_c_landscape_default(printer)
        printer.setDocName(result.output_path.stem)

        # Print the 81 x 51 mm card at its physical size, centered on the
        # selected C-size page. The selected media/orientation are deliberately
        # reapplied for every job because Canon driver preferences are per-PC.
        printer.setFullPage(True)
        page = printer.pageRect(QPrinter.Unit.DevicePixel)
        px_per_mm = printer.resolution() / 25.4
        card_w = 81.0 * px_per_mm
        card_h = 51.0 * px_per_mm
        if page.width() < card_w or page.height() < card_h:
            self._write_print_log(result, {
                "status": "rejected_page_too_small",
                "printer_name": printer.printerName(),
                "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
                "printer_resolution_dpi": printer.resolution(),
                "required_card_mm": [81.0, 51.0],
            })
            QMessageBox.warning(
                self, "용지 크기 확인",
                "선택한 용지가 8.1 × 5.1cm 인쇄 영역보다 작습니다.\n"
                "프린터 설정에서 SELPHY 카드 크기 또는 더 큰 용지를 선택하세요.",
            )
            return

        image = QImage(str(result.output_path))
        if image.isNull():
            self._write_print_log(result, {
                "status": "failed_to_load_image",
                "printer_name": printer.printerName(),
            })
            QMessageBox.critical(self, "인쇄 오류", "보정 이미지를 열 수 없습니다.")
            return
        target = QRectF(page.x() + (page.width() - card_w) / 2,
                        page.y() + (page.height() - card_h) / 2,
                        card_w, card_h)
        painter = QPainter()
        if not painter.begin(printer):
            state = printer.printerState()
            self._write_print_log(result, {
                "status": "painter_begin_failed",
                "printer_name": printer.printerName(),
                "printer_state": state.name,
            })
            QMessageBox.critical(self, "인쇄 오류", "프린터 작업을 시작하지 못했습니다.")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, image)
        ended = painter.end()
        state = printer.printerState()
        succeeded = ended and state != QPrinter.PrinterState.Aborted
        self._write_print_log(result, {
            "status": "sent_to_spooler" if succeeded else "print_job_failed",
            "printer_name": printer.printerName(),
            "printer_state": state.name,
            "painter_end": ended,
            "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
            "target_rect_px": [target.x(), target.y(), target.width(), target.height()],
            "printer_resolution_dpi": printer.resolution(),
            "page_size_mm": [86.0, 54.0],
            "media": "C card (54 x 86 mm)",
            "orientation": "landscape",
            "card_size_mm": [81.0, 51.0],
        })
        if not succeeded:
            QMessageBox.critical(
                self, "인쇄 오류",
                f"인쇄 작업 전송에 실패했습니다. 프린터 상태: {state.name}\n"
                "단계별 진단 폴더의 print_attempt.json을 확인하세요.",
            )
            return
        self.status.setText("인쇄 작업을 프린터로 보냈습니다. 프린터 드라이버에서 카드 용지와 테두리 설정을 확인하세요.")

    @staticmethod
    def _write_print_log(result: CropResult, info: dict) -> None:
        folder = result.debug_dir or (result.output_path.parent if result.output_path else Path.cwd())
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "print_attempt.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def open_output(self) -> None:
        folder = Path(self.output_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def set_busy(self, busy: bool) -> None:
        for button in (self.add_files_button, self.add_folder_button, self.remove_button,
                       self.process_selected_button, self.process_all_button,
                       self.manual_apply_button, self.perspective_mode):
            button.setEnabled(not busy)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("신분증 스캔 보정 및 인쇄")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
