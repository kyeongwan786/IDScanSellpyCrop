"""Desktop GUI for selecting, correcting, and printing ID card scans."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import cv2
from PySide6.QtCore import (Qt, QThread, Signal, QRect, QRectF, QUrl, QSize, QSizeF, QMarginsF, QSettings, QEvent,
                            QTimer,
                            QPoint, QMimeData, qVersion)
from PySide6.QtGui import (QColor, QDesktopServices, QDrag, QIcon, QImage, QImageReader,
                           QPalette,
                           QKeySequence, QPixmap, QShortcut,
                           QPageLayout, QPageSize, QPainter, QPdfWriter)
from PySide6.QtPrintSupport import QPrinter, QPrinterInfo
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStyle,
    QStyleOptionViewItem,
    QProxyStyle,
    QSizePolicy,
    QStackedWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .engine import CropResult, enhance_for_recognition, process_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
USER_ROLE = int(Qt.ItemDataRole.UserRole)
LIST_LABEL_ROLE = USER_ROLE + 1
SETTINGS_ORGANIZATION = "IDCardCropper"
SETTINGS_APPLICATION = "IDCardCropper"
DM_OUT_BUFFER = 0x00000002
DM_IN_PROMPT = 0x00000004
DM_IN_BUFFER = 0x00000008
IDOK = 1


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
            # QPageLayout applies Landscape by rotating its portrait base size.
            # Normalize driver entries that report C already as 86x54; otherwise
            # applying Landscape a second time can produce a 54x86 page.
            return QPageSize(QSizeF(54.0, 86.0), QPageSize.Unit.Millimeter,
                             supported_size.name())
    return QPageSize(QSizeF(54.0, 86.0), QPageSize.Unit.Millimeter, "C Size")


def _set_c_landscape_default(printer: QPrinter) -> None:
    printer.setPageSize(_c_size_for_printer(printer))
    printer.setPageOrientation(QPageLayout.Orientation.Landscape)


def _save_windows_printer_preferences(printer_name: str, parent_window: int) -> bool:
    """Show the driver's property sheet and save its DEVMODE as this user's default."""
    if sys.platform != "win32":
        raise OSError("Windows 프린터 드라이버 설정은 Windows에서만 저장할 수 있습니다.")

    import ctypes
    from ctypes import wintypes

    winspool = ctypes.WinDLL("winspool.drv", use_last_error=True)
    handle = wintypes.HANDLE()
    winspool.OpenPrinterW.argtypes = [
        wintypes.LPWSTR, ctypes.POINTER(wintypes.HANDLE), wintypes.LPVOID,
    ]
    winspool.OpenPrinterW.restype = wintypes.BOOL
    winspool.ClosePrinter.argtypes = [wintypes.HANDLE]
    winspool.ClosePrinter.restype = wintypes.BOOL
    winspool.DocumentPropertiesW.argtypes = [
        wintypes.HWND, wintypes.HANDLE, wintypes.LPWSTR,
        wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD,
    ]
    winspool.DocumentPropertiesW.restype = wintypes.LONG
    winspool.SetPrinterW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
    ]
    winspool.SetPrinterW.restype = wintypes.BOOL

    if not winspool.OpenPrinterW(printer_name, ctypes.byref(handle), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = winspool.DocumentPropertiesW(None, handle, printer_name, None, None, 0)
        if size <= 0:
            raise OSError("프린터 드라이버 설정 크기를 가져오지 못했습니다.")

        defaults = ctypes.create_string_buffer(size)
        result = winspool.DocumentPropertiesW(
            None, handle, printer_name, defaults, None, DM_OUT_BUFFER,
        )
        if result < 0:
            raise ctypes.WinError(ctypes.get_last_error())

        configured = ctypes.create_string_buffer(size)
        result = winspool.DocumentPropertiesW(
            wintypes.HWND(parent_window), handle, printer_name,
            configured, defaults, DM_OUT_BUFFER | DM_IN_PROMPT | DM_IN_BUFFER,
        )
        if result != IDOK:  # IDCANCEL leaves the saved defaults untouched.
            return False

        class PrinterInfo9(ctypes.Structure):
            _fields_ = [("pDevMode", wintypes.LPVOID)]

        info = PrinterInfo9(ctypes.cast(configured, wintypes.LPVOID))
        if not winspool.SetPrinterW(handle, 9, ctypes.byref(info), 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return True
    finally:
        winspool.ClosePrinter(handle)


class CheckablePhotoList(QListWidget):
    """Toggle an item's check state when its row body is clicked."""

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        point = event.position().toPoint()
        item = self.itemAt(point)
        toggle_item = None
        if (item and event.button() == Qt.MouseButton.LeftButton
                and item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            option = QStyleOptionViewItem()
            option.initFrom(self)
            option.rect = self.visualItemRect(item)
            option.features |= QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            option.checkState = item.checkState()
            indicator = self.style().subElementRect(
                QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, self,
            )
            if not indicator.contains(point):
                toggle_item = item

        super().mousePressEvent(event)
        if toggle_item:
            next_state = (
                Qt.CheckState.Unchecked
                if toggle_item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            toggle_item.setCheckState(next_state)


class ConsistentCheckStyle(QProxyStyle):
    """Use one checkbox drawing for both the list and the select-all control."""

    def subElementRect(self, element, option, widget=None):  # noqa: N802
        rect = super().subElementRect(element, option, widget)
        if element in (
            QStyle.SubElement.SE_CheckBoxIndicator,
            QStyle.SubElement.SE_ItemViewItemCheckIndicator,
        ):
            if rect.isValid() and not rect.isEmpty():
                center_x = rect.center().x()
                center_y = rect.center().y()
            else:
                center_x = option.rect.left() + 10
                center_y = option.rect.center().y()
            return QRect(center_x - 10, center_y - 10, 20, 20)
        return rect

    def drawPrimitive(self, element, option, painter, widget=None) -> None:  # noqa: N802
        if element in (
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
        ):
            if option.state & QStyle.StateFlag.State_NoChange:
                state = Qt.CheckState.PartiallyChecked
            elif option.state & QStyle.StateFlag.State_On:
                state = Qt.CheckState.Checked
            else:
                state = Qt.CheckState.Unchecked
            _paint_check_mark(painter, option.rect, state)
            return
        super().drawPrimitive(element, option, painter, widget)


class SelectAllCheckBox(QCheckBox):
    """Keep the select-all mark fully visible and aligned with list checks."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setMinimumHeight(26)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        painter = QPainter(self)
        rect = QRect(2, (self.height() - 20) // 2 + 2, 20, 20)
        _paint_check_mark(painter, rect, self.checkState())
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        if not self.isEnabled():
            text_color = QColor("#9aa4b4")
        painter.setPen(text_color)
        painter.drawText(
            QRect(30, 0, max(0, self.width() - 30), self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        painter.end()


def _paint_check_mark(painter, rect, check_state) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    checked = check_state == Qt.CheckState.Checked
    partial = check_state == Qt.CheckState.PartiallyChecked
    if checked:
        pen = painter.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setColor(QColor("#078b88"))
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 4, 4)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            QPoint(round(rect.left() + rect.width() * 0.23), round(rect.top() + rect.height() * 0.53)),
            QPoint(round(rect.left() + rect.width() * 0.43), round(rect.top() + rect.height() * 0.72)),
        )
        painter.drawLine(
            QPoint(round(rect.left() + rect.width() * 0.43), round(rect.top() + rect.height() * 0.72)),
            QPoint(round(rect.left() + rect.width() * 0.78), round(rect.top() + rect.height() * 0.30)),
        )
    else:
        pen = painter.pen()
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setColor(QColor("#58677d") if not partial else QColor("#078b88"))
        pen.setWidthF(1.8)
        painter.setPen(pen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 4, 4)
        if partial:
            pen = painter.pen()
            pen.setWidthF(2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(
                QPoint(round(rect.left() + rect.width() * 0.25), rect.center().y()),
                QPoint(round(rect.left() + rect.width() * 0.75), rect.center().y()),
            )
    painter.restore()


class Preview(QLabel):
    def __init__(self, title: str, *, allow_file_drag: bool = False) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("imagePreview")
        self.setStyleSheet(
            "QLabel { background: #f4f6f8; color: #778197; "
            "border: 1px dashed #ccd3dd; border-radius: 12px; padding: 12px; }"
        )
        self._pixmap: QPixmap | None = None
        self._file_path: Path | None = None
        self._allow_file_drag = allow_file_drag
        self._drag_origin: QPoint | None = None
        self._placeholder = title
        self._zoom = 1.0
        if allow_file_drag:
            self.setToolTip("마우스 휠로 확대·축소할 수 있습니다.")

    def show_image(self, path: Path | None) -> None:
        self._pixmap = QPixmap(str(path)) if path and path.is_file() else None
        self._file_path = Path(path).resolve() if self._pixmap and self._allow_file_drag else None
        self._zoom = 1.0
        self._render()

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._file_path = None
        self._zoom = 1.0
        self._render()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if not self._pixmap or self._pixmap.isNull():
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta:
            factor = 1.15 if delta > 0 else 1 / 1.15
            self._zoom = min(4.0, max(0.35, self._zoom * factor))
            self._render()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if (self._allow_file_drag and self._file_path
                and event.button() == Qt.MouseButton.LeftButton):
            self._drag_origin = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if (self._drag_origin is None or not self._file_path
                or not (event.buttons() & Qt.MouseButton.LeftButton)):
            return super().mouseMoveEvent(event)
        current = event.position().toPoint()
        if (current - self._drag_origin).manhattanLength() < QApplication.startDragDistance():
            return super().mouseMoveEvent(event)

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(self._file_path))])
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
                  Qt.DropAction.CopyAction)
        self._drag_origin = None

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self._allow_file_drag and self._file_path and event.button() == Qt.MouseButton.LeftButton:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._file_path)))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap and not self._pixmap.isNull():
            fit_size = QSize(max(1, self.width() - 28), max(1, self.height() - 28))
            fitted = self._pixmap.scaled(
                fit_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            zoomed_size = QSize(
                max(1, round(fitted.width() * self._zoom)),
                max(1, round(fitted.height() * self._zoom)),
            )
            self.setPixmap(fitted.scaled(
                zoomed_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)


class ProcessWorker(QThread):
    item_done = Signal(str, object)

    def __init__(self, paths: list[Path], output_dir: Path) -> None:
        super().__init__()
        self.paths = paths
        self.output_dir = output_dir

    def run(self) -> None:
        for path in self.paths:
            try:
                result = process_image(path, self.output_dir)
            except Exception as exc:
                # A bad path or unexpected pipeline exception must not terminate
                # the QThread before the UI receives a completion for this item.
                result = CropResult(
                    "failed",
                    f"이미지 처리 중 예외가 발생했습니다: {type(exc).__name__}: {exc}",
                )
            self.item_done.emit(str(path), result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("셀피 카드 스튜디오")
        self.resize(1240, 900)
        self.setMinimumSize(980, 740)
        self.setAcceptDrops(True)
        self.results: dict[str, CropResult] = {}
        self.recognition_outputs: dict[str, Path] = {}
        self.recognition_undo_stack: list[tuple[str, Path | None]] = []
        self.worker: ProcessWorker | None = None
        self.pending_processing: list[Path] = []
        self.pending_photoshop_paths: list[Path] = []
        self.photoshop_launch_pending = False
        self._active_print_attempt: dict | None = None
        self.last_print_log_error: str | None = None
        self._updating_selection = False
        self.is_busy = False

        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("셀피 카드 스튜디오")
        title.setObjectName("appTitle")
        subtitle = QLabel("사진을 모아 한 번에 보정하고 인쇄하세요")
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)
        header.addWidget(QLabel("저장 위치"))
        self.output_edit = QLineEdit(str(Path.cwd() / "output"))
        self.output_edit.setMinimumWidth(220)
        self.output_browse_button = QPushButton("찾아보기")
        header.addWidget(self.output_edit)
        header.addWidget(self.output_browse_button)
        layout.addLayout(header)

        steps = QLabel("1   사진 추가      ─────────      2   결과 확인      ─────────      3   선택 인쇄")
        steps.setObjectName("steps")
        steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(steps)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("workspaceSplitter")

        list_panel = QFrame()
        list_panel.setObjectName("panel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(16, 16, 16, 14)
        list_layout.setSpacing(10)
        list_header = QHBoxLayout()
        list_title = QLabel("사진 목록")
        list_title.setObjectName("sectionTitle")
        self.file_count = QLabel("0개")
        self.file_count.setObjectName("countBadge")
        list_header.addWidget(list_title)
        list_header.addWidget(self.file_count)
        list_header.addStretch(1)
        self.add_files_button = QPushButton("＋ 사진 추가")
        self.add_files_button.setObjectName("secondaryButton")
        list_header.addWidget(self.add_files_button)
        self.add_folder_button = QPushButton("폴더")
        self.add_folder_button.setObjectName("secondaryButton")
        list_header.addWidget(self.add_folder_button)
        list_layout.addLayout(list_header)

        self.select_all_checkbox = SelectAllCheckBox("모두 선택")
        self.select_all_checkbox.setTristate(True)
        list_layout.addWidget(self.select_all_checkbox)
        self.file_list = CheckablePhotoList()
        self.file_list.setObjectName("photoList")
        self._item_check_style = ConsistentCheckStyle()
        self.file_list.setStyle(self._item_check_style)
        self.file_list.setMinimumWidth(320)
        self.file_list.setIconSize(QSize(52, 52))
        self.file_list.setSpacing(6)
        list_layout.addWidget(self.file_list, 1)
        self.drop_hint = QLabel("사진을 여기에 끌어놓으세요")
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        list_layout.addWidget(self.drop_hint)
        list_actions = QHBoxLayout()
        self.remove_button = QPushButton("선택 삭제")
        self.remove_all_button = QPushButton("전체 삭제")
        list_actions.addWidget(self.remove_button)
        list_actions.addWidget(self.remove_all_button)
        list_layout.addLayout(list_actions)
        splitter.addWidget(list_panel)

        preview_panel = QFrame()
        preview_panel.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(18, 16, 18, 14)
        preview_layout.setSpacing(12)
        preview_header = QHBoxLayout()
        preview_title = QLabel("미리보기")
        preview_title.setObjectName("sectionTitle")
        self.preview_filename = QLabel("사진을 선택하세요")
        self.preview_filename.setObjectName("mutedLabel")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_filename)
        preview_layout.addLayout(preview_header)

        tabs = QHBoxLayout()
        self.source_tab = QPushButton("원본")
        self.result_tab = QPushButton("보정 결과")
        for tab in (self.source_tab, self.result_tab):
            tab.setObjectName("tabButton")
            tab.setCheckable(True)
            tabs.addWidget(tab)
        tabs.addStretch(1)
        preview_layout.addLayout(tabs)

        self.preview_stack = QStackedWidget()
        self.source_preview = Preview("원본 미리보기")
        self.result_preview = Preview("보정 결과 미리보기", allow_file_drag=True)
        self.preview_stack.addWidget(self.source_preview)
        self.preview_stack.addWidget(self.result_preview)
        preview_layout.addWidget(self.preview_stack, 1)

        preview_meta = QHBoxLayout()
        self.preview_size = QLabel("8.1 × 5.1 cm · 원본 해상도 유지")
        self.preview_size.setObjectName("mutedLabel")
        self.fill_note = QLabel("이미지 꽉 채움")
        self.fill_note.setObjectName("successBadge")
        preview_meta.addWidget(self.preview_size)
        preview_meta.addStretch(1)
        preview_meta.addWidget(self.fill_note)
        preview_layout.addLayout(preview_meta)

        preview_tools = QHBoxLayout()
        self.recognition_enhance_button = QPushButton("인식 보정")
        self.recognition_enhance_button.setToolTip(
            "현재 사진의 색과 밝기를 보정합니다. 크롭은 하지 않습니다."
        )
        self.undo_button = QPushButton("되돌리기")
        self.undo_button.setToolTip("마지막 인식 보정을 되돌립니다 (Ctrl+Z).")
        self.open_diagnostics_button = QPushButton("단계별 진단")
        self.open_output_button = QPushButton("결과 폴더 열기")
        self.open_print_logs_button = QPushButton("인쇄 로그 열기")
        preview_tools.addWidget(self.recognition_enhance_button)
        preview_tools.addWidget(self.undo_button)
        preview_tools.addStretch(1)
        preview_tools.addWidget(self.open_diagnostics_button)
        preview_tools.addWidget(self.open_output_button)
        preview_tools.addWidget(self.open_print_logs_button)
        preview_layout.addLayout(preview_tools)

        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 810])
        layout.addWidget(splitter, 1)

        footer = QFrame()
        footer.setObjectName("actionBar")
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(14, 10, 14, 10)
        self.process_selected_button = QPushButton("선택 항목 보정")
        self.process_selected_button.setObjectName("secondaryButton")
        self.clear_selection_button = QPushButton("선택 해제")
        self.selected_count = QLabel("0개 선택됨")
        self.selected_count.setObjectName("selectedCount")
        actions.addWidget(self.selected_count)
        actions.addSpacing(12)
        actions.addWidget(self.clear_selection_button)
        actions.addWidget(self.process_selected_button)
        self.process_all_button = QPushButton("전체 보정")
        self.process_all_button.setObjectName("secondaryButton")
        actions.addWidget(self.process_all_button)
        actions.addStretch(1)
        self.print_button = QPushButton("선택 항목 인쇄")
        self.print_button.setObjectName("primaryButton")
        self.print_button.setMinimumWidth(170)
        self.print_all_button = QPushButton("전체 인쇄")
        self.print_all_button.setObjectName("primaryButton")
        self.print_all_button.setMinimumWidth(140)
        self.edit_selected_button = QPushButton("수정하기")
        self.edit_selected_button.setObjectName("secondaryButton")
        self.edit_selected_button.setToolTip(
            "체크한 사진을 목록 순서대로 Photoshop에서 엽니다."
        )
        actions.addWidget(self.print_button)
        actions.addWidget(self.print_all_button)
        actions.addWidget(self.edit_selected_button)
        layout.addWidget(footer)

        self.status = QLabel("사진을 추가하거나 이 창으로 끌어오세요.")
        self.status.setObjectName("statusLine")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.setStyleSheet("""
            QMainWindow, QWidget#appRoot { background: #f4f6f9; color: #17233c; }
            QWidget { font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; font-size: 10pt; }
            QFrame#panel { background: #ffffff; border: 1px solid #e0e5ec; border-radius: 14px; }
            QFrame#actionBar { background: #ffffff; border: 1px solid #e0e5ec; border-radius: 12px; }
            QLabel#appTitle { font-size: 20pt; font-weight: 700; color: #14213d; }
            QLabel#appSubtitle, QLabel#mutedLabel, QLabel#statusLine { color: #69758b; }
            QLabel#steps { color: #53627a; background: #eaf1f3; border-radius: 10px; padding: 9px; }
            QLabel#sectionTitle { font-size: 14pt; font-weight: 700; color: #17233c; }
            QLabel#countBadge, QLabel#successBadge { color: #087e80; background: #e4f4f1; border-radius: 9px; padding: 5px 9px; }
            QLabel#selectedCount { font-size: 12pt; font-weight: 700; color: #087e80; }
            QLabel#dropHint { color: #778197; border: 1px dashed #c8d0dc; border-radius: 10px; padding: 12px; }
            QLineEdit { background: #ffffff; border: 1px solid #d6dde7; border-radius: 8px; padding: 8px 10px; color: #17233c; }
            QListWidget#photoList { background: transparent; border: none; outline: none; }
            QListWidget#photoList::item { border: 1px solid #e1e6ed; border-radius: 9px; padding: 13px 9px; margin: 1px 0; }
            QListWidget#photoList::item:hover { border-color: #98c9c6; }
            QListWidget#photoList::item:selected { background: #ffffff; border-color: #078b88; color: #34435d; }
            QCheckBox { color: #34435d; spacing: 8px; }
            QPushButton { background: #ffffff; color: #34435d; border: 1px solid #d6dde7; border-radius: 8px; padding: 8px 12px; }
            QPushButton:hover { background: #f2f7f7; border-color: #94c9c5; }
            QPushButton:disabled { color: #9aa4b4; background: #f0f2f5; border-color: #e2e6ec; }
            QPushButton#secondaryButton { background: #ffffff; }
            QPushButton#primaryButton { color: #ffffff; background: #078b88; border: 1px solid #078b88; font-weight: 700; padding: 11px 16px; }
            QPushButton#primaryButton:hover { background: #067775; }
            QPushButton#tabButton { background: #f1f3f6; border: none; border-radius: 8px; padding: 8px 18px; }
            QPushButton#tabButton:checked { color: #087e80; background: #e2f3f0; font-weight: 700; }
        """)

        self.add_files_button.clicked.connect(self.add_files)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.remove_all_button.clicked.connect(self.remove_all)
        self.output_browse_button.clicked.connect(self.choose_output)
        self.process_selected_button.clicked.connect(self.process_selected)
        self.process_all_button.clicked.connect(self.process_all)
        self.recognition_enhance_button.clicked.connect(self.enhance_selected_for_recognition)
        self.undo_button.clicked.connect(self.undo_last_recognition_enhancement)
        self.file_list.currentItemChanged.connect(self.selection_changed)
        self.file_list.itemChanged.connect(self.selection_checks_changed)
        self.select_all_checkbox.toggled.connect(self.set_all_selected)
        self.clear_selection_button.clicked.connect(self.clear_selection)
        self.source_tab.clicked.connect(lambda: self.set_preview_mode("source"))
        self.result_tab.clicked.connect(lambda: self.set_preview_mode("result"))
        self.print_button.clicked.connect(
            lambda: self.handle_print_clicked(self.selected_paths(), "선택한"),
        )
        self.print_all_button.clicked.connect(
            lambda: self.handle_print_clicked(self.all_paths(), "전체"),
        )
        self.edit_selected_button.clicked.connect(self.edit_selected_images)
        self.open_diagnostics_button.clicked.connect(self.open_diagnostics)
        self.open_output_button.clicked.connect(self.open_output)
        self.open_print_logs_button.clicked.connect(self.open_print_logs)
        self.open_diagnostics_button.setEnabled(False)
        self.recognition_enhance_button.setEnabled(False)
        self.source_tab.setChecked(True)
        self.update_selection_controls()
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_recognition_enhancement)
        self._update_undo_controls()
        app = QApplication.instance()
        if app:
            app.focusChanged.connect(self._update_undo_controls)

        # Qt delivers drag events to the child widget under the pointer. Install
        # the same file-drop handler on the full UI so buttons, previews, and
        # the list all accept files just like the unused background does.
        for widget in [root, *root.findChildren(QWidget)]:
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)

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
        added_paths: list[Path] = []
        for path in paths:
            resolved = str(path.resolve())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and resolved not in known:
                item = QListWidgetItem()
                item.setData(USER_ROLE, resolved)
                item.setData(LIST_LABEL_ROLE, f"대기 중   ·   {path.name}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setIcon(QIcon(resolved))
                item.setToolTip(resolved)
                self.file_list.addItem(item)
                self._style_list_item(item)
                known.add(resolved)
                added_paths.append(Path(resolved))
        if added_paths:
            self.file_list.setCurrentRow(self.file_list.count() - len(added_paths))
            self.status.setText(
                f"{len(added_paths)}개 사진을 추가했습니다. 인쇄·수정할 사진은 체크하세요."
            )
        self.drop_hint.setVisible(self.file_list.count() == 0)
        self.update_selection_controls()

    def _style_list_item(self, item: QListWidgetItem) -> None:
        label = item.data(LIST_LABEL_ROLE) or ""
        old_blocked = self.file_list.blockSignals(True)
        try:
            display_text = label
            if item.text() != display_text:
                item.setText(display_text)
            background = QColor("#ffffff")
            foreground = QColor("#34435d")
            if item.background().color() != background:
                item.setBackground(background)
            if item.foreground().color() != foreground:
                item.setForeground(foreground)
        finally:
            self.file_list.blockSignals(old_blocked)

    def _set_list_item_label(self, item: QListWidgetItem, label: str) -> None:
        item.setData(LIST_LABEL_ROLE, label)
        self._style_list_item(item)

    @staticmethod
    def _dropped_image_paths(mime_data: QMimeData) -> list[Path]:
        paths = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                paths.append(path)
        return paths

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self._dropped_image_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self._dropped_image_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        paths = self._dropped_image_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        self.add_paths(paths)
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt callback name
        event_type = event.type()
        if event_type in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if self._dropped_image_paths(event.mimeData()):
                event.acceptProposedAction()
            else:
                event.ignore()
            return True
        if event_type == QEvent.Type.Drop:
            paths = self._dropped_image_paths(event.mimeData())
            if paths:
                self.add_paths(paths)
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return True
        return super().eventFilter(watched, event)

    def remove_selected(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        rows = [i for i in range(self.file_list.count())
                if self.file_list.item(i).checkState() == Qt.CheckState.Checked]
        if not rows:
            self.status.setText("목록에서 제거할 사진을 먼저 선택하세요.")
            return
        self.file_list.blockSignals(True)
        try:
            for row in reversed(rows):
                item = self.file_list.takeItem(row)
                raw_path = item.data(USER_ROLE)
                self.results.pop(raw_path, None)
                self.recognition_outputs.pop(raw_path, None)
                self._discard_undo_for(raw_path)
        finally:
            self.file_list.blockSignals(False)
        self.drop_hint.setVisible(self.file_list.count() == 0)
        if self.file_list.count():
            self.file_list.setCurrentRow(min(rows[0], self.file_list.count() - 1))
            self.selection_changed(self.file_list.currentItem(), None)
        else:
            self.selection_changed(None, None)
        self.update_selection_controls()
        self.status.setText(f"{len(rows)}개 사진을 목록에서 제거했습니다.")

    def remove_all(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        total = self.file_list.count()
        if not total:
            return
        answer = QMessageBox.question(
            self, "목록 전체 삭제",
            f"목록의 사진 {total}개를 모두 제거할까요?\n원본 파일은 삭제되지 않습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.file_list.blockSignals(True)
        try:
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                raw_path = item.data(USER_ROLE)
                self.results.pop(raw_path, None)
                self.recognition_outputs.pop(raw_path, None)
                self._discard_undo_for(raw_path)
            self.file_list.clear()
        finally:
            self.file_list.blockSignals(False)
        self.drop_hint.setVisible(True)
        self.selection_changed(None, None)
        self.update_selection_controls()
        self.status.setText(f"목록에서 사진 {total}개를 모두 제거했습니다. 원본 파일은 그대로 있습니다.")

    def choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "결과 폴더 선택", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def selected_path(self) -> Path | None:
        item = self.file_list.currentItem()
        return Path(item.data(USER_ROLE)) if item else None

    def selected_paths(self) -> list[Path]:
        return [
            Path(self.file_list.item(i).data(USER_ROLE))
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def all_paths(self) -> list[Path]:
        return [
            Path(self.file_list.item(i).data(USER_ROLE))
            for i in range(self.file_list.count())
        ]

    def set_all_selected(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating_selection = True
        try:
            for i in range(self.file_list.count()):
                self.file_list.item(i).setCheckState(state)
        finally:
            self._updating_selection = False
        self.update_selection_controls()

    def clear_selection(self) -> None:
        self.set_all_selected(False)
        self.status.setText("사진 선택을 해제했습니다.")

    def selection_checks_changed(self, _item: QListWidgetItem) -> None:
        self._style_list_item(_item)
        if self._updating_selection:
            return
        self.update_selection_controls()

    def _discard_undo_for(self, raw_path: str) -> None:
        self.recognition_undo_stack = [
            entry for entry in self.recognition_undo_stack if entry[0] != raw_path
        ]
        self._update_undo_controls()

    def update_selection_controls(self) -> None:
        total = self.file_list.count()
        selected = self.selected_paths()
        count = len(selected)
        self.file_count.setText(f"{total}개")
        self.selected_count.setText(f"{count}개 선택됨")

        busy = self.is_busy
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setEnabled(total > 0 and not busy)
        if count == 0:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        elif count == total:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Checked)
        else:
            self.select_all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self.select_all_checkbox.blockSignals(False)

        self.process_selected_button.setEnabled(count > 0 and not busy)
        self.process_all_button.setEnabled(total > 0 and not busy)
        self.clear_selection_button.setEnabled(count > 0 and not busy)
        self.remove_button.setEnabled(count > 0 and not busy)
        self.remove_all_button.setEnabled(total > 0 and not busy)
        ready_to_print = count > 0 and all(self._printable_path(path) for path in selected)
        self.print_button.setEnabled(ready_to_print and not busy)
        all_paths = self.all_paths()
        all_ready = bool(total) and all(self._printable_path(path) for path in all_paths)
        self.print_all_button.setEnabled(all_ready and not busy)
        self.edit_selected_button.setEnabled(count > 0 and not busy and not self.photoshop_launch_pending)
        if busy:
            self.print_button.setToolTip("보정이 끝난 뒤 선택한 사진을 한 번에 인쇄할 수 있습니다.")
            self.print_all_button.setToolTip("보정이 끝난 뒤 목록의 모든 사진을 순서대로 인쇄합니다.")
        elif not count:
            self.print_button.setToolTip("인쇄할 사진을 선택하세요.")
        elif not ready_to_print:
            self.print_button.setToolTip("선택한 사진을 먼저 모두 보정하세요.")
        else:
            self.print_button.setToolTip(f"선택한 {count}개 사진을 한 번에 인쇄합니다.")
        if not busy and not total:
            self.print_all_button.setToolTip("먼저 사진을 목록에 추가하세요.")
        elif not busy and not all_ready:
            self.print_all_button.setToolTip("목록의 사진을 모두 보정한 뒤 인쇄할 수 있습니다.")
        elif not busy:
            self.print_all_button.setToolTip(f"목록의 {total}개 사진을 순서대로 인쇄합니다.")

    def _printable_path(self, path: Path) -> Path | None:
        raw_path = str(path)
        recognition_path = self.recognition_outputs.get(raw_path)
        if recognition_path and recognition_path.is_file():
            return recognition_path
        result = self.results.get(raw_path)
        if (result and result.status == "complete" and result.output_path
                and result.output_path.is_file()):
            return result.output_path
        return None

    def set_preview_mode(self, mode: str) -> None:
        is_result = mode == "result"
        self.preview_stack.setCurrentWidget(
            self.result_preview if is_result else self.source_preview
        )
        self.result_tab.setChecked(is_result)
        self.source_tab.setChecked(not is_result)

    def process_selected(self) -> None:
        paths = self.selected_paths()
        if not paths:
            self.status.setText("보정할 사진을 선택하세요.")
            return
        self.start_processing(paths)

    def process_all(self) -> None:
        paths = self.all_paths()
        if not paths:
            self.status.setText("보정할 사진이 목록에 없습니다.")
            return
        self.start_processing(paths)

    def start_processing(self, paths: list[Path]) -> None:
        if self.worker and self.worker.isRunning():
            self.pending_processing.extend(paths)
            return
        self.set_preview_mode("result")
        for path in paths:
            self.recognition_outputs.pop(str(path), None)
            self._discard_undo_for(str(path))
            for index in range(self.file_list.count()):
                item = self.file_list.item(index)
                if item.data(USER_ROLE) == str(path):
                    self._set_list_item_label(item, f"처리 중   ·   {path.name}")
                    break
        output_dir = Path(self.output_edit.text()).expanduser()
        self.set_busy(True)
        self.status.setText(
            f"{len(paths)}개 이미지의 기본 보정을 처리하고 있습니다…"
        )
        self.worker = ProcessWorker(paths, output_dir)
        self.worker.item_done.connect(self.processing_done)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
        self.update_selection_controls()

    def processing_finished(self) -> None:
        self.set_busy(False)
        if self.pending_processing:
            queued = self.pending_processing
            self.pending_processing = []
            self.start_processing(queued)

    def processing_done(self, raw_path: str, result: CropResult) -> None:
        self.results[raw_path] = result
        completed_item = None
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(USER_ROLE) == raw_path:
                completed_item = item
                self._set_list_item_label(
                    item, f"{self._status_text(result.status)}   ·   {Path(raw_path).name}",
                )
                break
        # Follow the worker's completion order so the preview always shows the
        # result that just finished, even during a multi-photo batch.
        self.set_preview_mode("result")
        if completed_item and self.file_list.currentItem() is not completed_item:
            self.file_list.setCurrentItem(completed_item)
        else:
            self.display_path(Path(raw_path))
        message = f"{Path(raw_path).name} · {result.message}"
        if result.quality_warning:
            message += " " + result.quality_warning
        if result.debug_dir:
            message += f" 단계별 캡처 저장: {result.debug_dir}"
        self.status.setText(message)
        self.update_selection_controls()

    @staticmethod
    def _status_text(status: str) -> str:
        return {"complete": "완료", "manual_review": "수동 확인", "failed": "실패"}.get(status, status)

    def selection_changed(self, current: QListWidgetItem | None, _previous) -> None:
        path = Path(current.data(USER_ROLE)) if current else None
        if path:
            self.display_path(path)
        else:
            self.preview_filename.setText("사진을 선택하세요")
            self.source_preview.show_image(None)
            self.result_preview.show_image(None)
            self.open_diagnostics_button.setEnabled(False)
            self.recognition_enhance_button.setEnabled(False)

    def display_path(self, path: Path) -> None:
        self.preview_filename.setText(path.name)
        self.source_preview.show_image(path)
        raw_path = str(path)
        result = self.results.get(raw_path)
        recognition_path = self.recognition_outputs.get(raw_path)
        if recognition_path and recognition_path.is_file():
            self.result_preview.show_image(recognition_path)
        elif result:
            self.result_preview.show_image(result.output_path or result.preview_path)
        else:
            self.result_preview.show_image(None)
        if result:
            self.open_diagnostics_button.setEnabled(bool(result.debug_dir and result.debug_dir.is_dir()))
        else:
            self.open_diagnostics_button.setEnabled(False)
        busy = self.is_busy
        self.recognition_enhance_button.setEnabled(path.is_file() and not busy)
        self.update_selection_controls()

    @staticmethod
    def _recognition_output_path(input_path: Path, output_dir: Path) -> Path:
        candidate = output_dir / f"{input_path.stem}_인식보정.jpg"
        version = 2
        while candidate.exists():
            candidate = output_dir / f"{input_path.stem}_인식보정_{version}.jpg"
            version += 1
        return candidate

    def enhance_selected_for_recognition(self) -> None:
        source_path = self.selected_path()
        if not source_path or not source_path.is_file():
            QMessageBox.information(self, "원본 이미지 없음", "먼저 이미지를 추가하고 선택하세요.")
            return
        crop_result = self.results.get(str(source_path))
        enhancement_input = (
            crop_result.output_path
            if crop_result and crop_result.status == "complete"
            and crop_result.output_path and crop_result.output_path.is_file()
            else source_path
        )
        try:
            output_dir = Path(self.output_edit.text()).expanduser()
            enhanced_path = self._recognition_output_path(enhancement_input, output_dir)
            _, details = enhance_for_recognition(
                enhancement_input, enhanced_path,
            )
            details_path = enhanced_path.with_suffix(".json")
            details_path.write_text(
                json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            raw_path = str(source_path)
            previous_output = self.recognition_outputs.get(raw_path)
            self.recognition_undo_stack.append((raw_path, previous_output))
            self.recognition_outputs[raw_path] = enhanced_path
            self._update_undo_controls()
            self.display_path(source_path)
            before_metrics = details["metrics_before"]
            after_metrics = details["metrics_after"]
            before_l = before_metrics["lightness_p10_p90"]
            after_l = after_metrics["lightness_p10_p90"]
            before_median = before_metrics["lightness_p01_p50_p99"][1]
            after_median = after_metrics["lightness_p01_p50_p99"][1]
            adjustments = (
                f"밝기 p10/p50/p90 "
                f"{before_l[0]:.0f}/{before_median:.0f}/{before_l[1]:.0f}→"
                f"{after_l[0]:.0f}/{after_median:.0f}/{after_l[1]:.0f}, "
                f"채도 평균 {before_metrics['saturation_mean']:.0f}→"
                f"{after_metrics['saturation_mean']:.0f}"
            )
            step_names = {
                "paired_levels_curve": "비포·애프터 기반 레벨 곡선",
                "paired_chroma_gain": "비포·애프터 기반 채도 조정",
                "quality_already_adequate_noop": "이미지 상태 양호, 원본 유지",
            }
            actions = ", ".join(step_names.get(step, step)
                                 for step in details.get("actions", []))
            self.status.setText(
                f"인식 보정본을 만들었습니다: {enhanced_path.name} "
                f"(입력: {'크롭 결과' if enhancement_input != source_path else '원본'}; "
                f"DPI 표기 {details['output_dpi']}dpi, 추가 크롭·리사이즈 없이 입력 픽셀 크기 "
                f"{details['output_size_px'][0]}×"
                f"{details['output_size_px'][1]}px 유지; 적용: {actions}; {adjustments}). "
                "인쇄 버튼은 이 인식 보정본을 사용합니다."
            )
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "인식 보정 실패", f"이미지를 추가 보정하지 못했습니다.\n{exc}")

    def undo_last_recognition_enhancement(self) -> None:
        if not self.recognition_undo_stack:
            return
        raw_path, previous_output = self.recognition_undo_stack.pop()
        if previous_output and previous_output.is_file():
            self.recognition_outputs[raw_path] = previous_output
        else:
            self.recognition_outputs.pop(raw_path, None)
        source_path = Path(raw_path)
        for index in range(self.file_list.count()):
            if self.file_list.item(index).data(USER_ROLE) == raw_path:
                self.file_list.setCurrentRow(index)
                break
        self.display_path(source_path)
        self.set_preview_mode("result")
        restored = previous_output.name if previous_output and previous_output.is_file() else "이전 크롭 결과/원본"
        self.status.setText(f"마지막 인식 보정을 되돌렸습니다. 현재 사용 결과: {restored}")
        self._update_undo_controls()

    def _update_undo_controls(self, *_args) -> None:
        can_undo = bool(self.recognition_undo_stack) and not self.is_busy
        self.undo_button.setEnabled(can_undo)
        focused = QApplication.focusWidget()
        self.undo_shortcut.setEnabled(can_undo and not isinstance(focused, QLineEdit))

    def open_diagnostics(self) -> None:
        path = self.selected_path()
        result = self.results.get(str(path)) if path else None
        if not result or not result.debug_dir or not result.debug_dir.is_dir():
            QMessageBox.information(self, "진단 자료 없음", "먼저 이미지를 보정하세요.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.debug_dir.resolve())))

    @staticmethod
    def _runtime_print_diagnostics() -> dict:
        module_path = Path(__file__).resolve()
        executable_path = Path(sys.executable).resolve()

        def file_info(path: Path) -> dict:
            try:
                stat = path.stat()
                info = {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
                        timespec="seconds"
                    ),
                }
                if path.suffix.lower() == ".py" and stat.st_size <= 2_000_000:
                    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                return info
            except OSError as exc:
                return {"path": str(path), "exists": False,
                        "error": f"{type(exc).__name__}: {exc}"}

        repo_root = module_path.parents[2]
        git_commit = None
        if (repo_root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=repo_root,
                    capture_output=True, text=True, timeout=2, check=False,
                )
                if result.returncode == 0:
                    git_commit = result.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            from importlib.metadata import PackageNotFoundError, version
            try:
                package_version = version("id-card-scan-cropper")
            except PackageNotFoundError:
                package_version = None
        except ImportError:
            package_version = None
        app_info = {
            "package_version": package_version,
            "git_commit": git_commit,
            "frozen_executable": bool(getattr(sys, "frozen", False)),
            "argv0": sys.argv[0] if sys.argv else None,
            "executable": file_info(executable_path),
            "gui_module": file_info(module_path),
            "pyinstaller_extract_dir": getattr(sys, "_MEIPASS", None),
        }
        return {
            "application": app_info,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "platform_string": platform.platform(),
                "python": sys.version,
                "qt": qVersion(),
            },
        }

    def _start_print_attempt(self, scope_label: str, paths: list[Path]) -> Path | None:
        self.last_print_log_error = None
        output_dir = Path(self.output_edit.text()).expanduser()
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        log_path = output_dir / "print_logs" / f"print_{stamp}_{uuid4().hex[:8]}.json"
        self._active_print_attempt = {
            "path": log_path,
            "write_error": None,
            "data": {
                "log_version": 2,
                "attempt_id": log_path.stem,
                "started_at": timestamp,
                **self._runtime_print_diagnostics(),
                "scope": scope_label,
                "requested_count": len(paths),
                "requested_files": [
                    {"order": index + 1, "source_path": str(path.resolve())}
                    for index, path in enumerate(paths)
                ],
                "files": [],
                "pages": [],
                "events": [],
                "physical_print_confirmed": False,
            },
        }
        self._record_print_event("attempt_started")
        return log_path

    def _save_print_attempt_log(self) -> None:
        attempt = self._active_print_attempt
        if not attempt or attempt["write_error"]:
            return
        try:
            path: Path = attempt["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(attempt["data"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            attempt["write_error"] = f"{type(exc).__name__}: {exc}"

    def _record_print_event(self, event: str, **details) -> None:
        attempt = self._active_print_attempt
        if not attempt:
            return
        attempt["data"]["events"].append({
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **details,
        })
        self._save_print_attempt_log()

    @staticmethod
    def _rect_diagnostics(rect: QRectF | QRect) -> dict[str, float]:
        return {
            "x": float(rect.x()),
            "y": float(rect.y()),
            "width": float(rect.width()),
            "height": float(rect.height()),
            "left": float(rect.left()),
            "top": float(rect.top()),
            "right": float(rect.right()),
            "bottom": float(rect.bottom()),
        }

    @staticmethod
    def _print_image_diagnostics(image: QImage, path: Path) -> dict:
        """Capture image identity and edge pixels before handing it to a driver."""
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digest_error = None
        except OSError as exc:
            digest = None
            digest_error = f"{type(exc).__name__}: {exc}"

        width = image.width()
        height = image.height()
        sample_x = sorted({0, 1, 2, 3, 4, 8, 16, width // 4, width // 2,
                           (width * 3) // 4, max(0, width - 17),
                           max(0, width - 9), max(0, width - 5),
                           max(0, width - 4), max(0, width - 3),
                           max(0, width - 2), max(0, width - 1)})
        sample_y = sorted({0, 1, 2, 3, 4, 8, 16, height // 4, height // 2,
                           (height * 3) // 4, max(0, height - 17),
                           max(0, height - 9), max(0, height - 5),
                           max(0, height - 4), max(0, height - 3),
                           max(0, height - 2), max(0, height - 1)})
        edge_samples = {}
        for edge, y in (("top", 0), ("bottom", max(0, height - 1))):
            edge_samples[edge] = [
                {
                    "x": x,
                    "rgba": image.pixelColor(x, y).getRgb(),
                }
                for x in sample_x
            ]
        for edge, x in (("left", 0), ("right", max(0, width - 1))):
            edge_samples[edge] = [
                {
                    "y": y,
                    "rgba": image.pixelColor(x, y).getRgb(),
                }
                for y in sample_y
            ]

        # Summarize narrow edge bands, where a one- or two-pixel white seam
        # can hide between the point samples above. Cap long-edge sampling to
        # roughly 512 positions so larger scans do not stall printing.
        band_stats = {}
        for side in ("left", "right", "top", "bottom"):
            horizontal_edge = side in ("top", "bottom")
            span = width if horizontal_edge else height
            stride = max(1, span // 512)
            side_stats = {"sample_stride_px": stride, "bands": {}}
            for band_width in (1, 2, 4, 8, 16):
                band = min(band_width, height if horizontal_edge else width)
                channels = [0, 0, 0, 0]
                minimum = [255, 255, 255, 255]
                maximum = [0, 0, 0, 0]
                white_count = 0
                translucent_count = 0
                count = 0
                for position in range(0, span, stride):
                    for depth in range(band):
                        if side == "left":
                            x, y = depth, position
                        elif side == "right":
                            x, y = width - band + depth, position
                        elif side == "top":
                            x, y = position, depth
                        else:
                            x, y = position, height - band + depth
                        rgba = image.pixelColor(x, y).getRgb()
                        for channel in range(4):
                            channels[channel] += rgba[channel]
                            minimum[channel] = min(minimum[channel], rgba[channel])
                            maximum[channel] = max(maximum[channel], rgba[channel])
                        white_count += all(value >= 250 for value in rgba[:3])
                        translucent_count += rgba[3] < 255
                        count += 1
                side_stats["bands"][str(band_width)] = {
                    "sampled_pixels": count,
                    "mean_rgba": [round(value / count, 3) for value in channels],
                    "min_rgba": minimum,
                    "max_rgba": maximum,
                    "near_white_fraction": round(white_count / count, 6),
                    "non_opaque_fraction": round(translucent_count / count, 6),
                }
            band_stats[side] = side_stats

        return {
            "path": str(path.resolve()),
            "sha256": digest,
            "sha256_error": digest_error,
            "size_px": [width, height],
            "format": image.format().name,
            "bytes_per_line": image.bytesPerLine(),
            "device_pixel_ratio": image.devicePixelRatio(),
            "dots_per_meter": [image.dotsPerMeterX(), image.dotsPerMeterY()],
            "has_alpha_channel": image.hasAlphaChannel(),
            "color_space_valid": image.colorSpace().isValid(),
            "color_space_description": image.colorSpace().description(),
            "edge_samples_rgba": edge_samples,
            "edge_band_statistics": band_stats,
        }

    @classmethod
    def _write_print_geometry_preview(
        cls,
        path: Path,
        image: QImage,
        target: QRectF,
        paper: QRect,
        page: QRect,
        full_page: bool,
    ) -> dict:
        """Save the app-side raster as placed for printing, before driver handling."""
        preview = QImage(
            max(1, paper.width()), max(1, paper.height()),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        preview.fill(Qt.GlobalColor.white)
        painter = QPainter(preview)
        if not painter.isActive():
            raise RuntimeError("Could not start diagnostic preview painter")
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not full_page:
            painter.translate(page.x(), page.y())
            painter.setClipRect(
                QRectF(0, 0, page.width(), page.height()),
                Qt.ClipOperation.ReplaceClip,
            )
        painter.drawImage(target, image)
        painter.end()
        if not preview.save(str(path), "PNG"):
            raise OSError(f"Could not save diagnostic preview: {path}")
        preview_diagnostics = cls._print_image_diagnostics(preview, path)
        return {
            "path": str(path.resolve()),
            "sha256": preview_diagnostics["sha256"],
            "stage": "Qt-side raster after app placement/clipping, before OS printer driver",
            "size_px": [preview.width(), preview.height()],
            "paper_rect_device_px": [paper.x(), paper.y(), paper.width(), paper.height()],
            "page_rect_device_px": [page.x(), page.y(), page.width(), page.height()],
            "full_page": full_page,
            "edge_samples_rgba": preview_diagnostics["edge_samples_rgba"],
            "edge_band_statistics": preview_diagnostics["edge_band_statistics"],
        }

    def _finish_print_attempt(self, status: str, **details) -> Path | None:
        attempt = self._active_print_attempt
        if not attempt:
            return None
        data = attempt["data"]
        data["status"] = status
        data["finished_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        data.update(details)
        self._record_print_event("attempt_finished", status=status, **details)
        self._save_print_attempt_log()
        if attempt["write_error"]:
            data["log_write_error"] = attempt["write_error"]
            self.last_print_log_error = attempt["write_error"]
        path = attempt["path"]
        self._active_print_attempt = None
        return path

    @staticmethod
    def _export_virtual_print_pdf(
        path: Path,
        items: list[tuple[Path, CropResult | None, Path]],
        photo_size_mm: list[float],
        margins_mm: list[float],
        dpi: int,
    ) -> None:
        """Render the exact printer target onto virtual 86 x 54 mm PDF pages."""
        writer = QPdfWriter(str(path))
        writer.setResolution(max(72, dpi))
        page_size = QPageSize(
            QSizeF(54.0, 86.0), QPageSize.Unit.Millimeter, "SELPHY C 86 x 54 mm",
        )
        layout = QPageLayout(
            page_size,
            QPageLayout.Orientation.Landscape,
            QMarginsF(0.0, 0.0, 0.0, 0.0),
            QPageLayout.Unit.Millimeter,
            QMarginsF(0.0, 0.0, 0.0, 0.0),
        )
        if not writer.setPageLayout(layout):
            raise OSError("가상 PDF에 86 × 54mm 페이지 크기를 설정하지 못했습니다.")

        page = writer.pageLayout().fullRectPixels(writer.resolution())
        px_per_mm_x = page.width() / 86.0
        px_per_mm_y = page.height() / 54.0
        target = QRectF(
            page.x() + margins_mm[0] * px_per_mm_x,
            page.y() + margins_mm[1] * px_per_mm_y,
            photo_size_mm[0] * px_per_mm_x,
            photo_size_mm[1] * px_per_mm_y,
        )

        painter = QPainter()
        if not painter.begin(writer):
            raise OSError("가상 PDF 파일을 만들지 못했습니다.")
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            for index, (_, _, image_path) in enumerate(items):
                image = QImage(str(image_path))
                if image.isNull():
                    raise OSError(f"PDF 검사 이미지를 열 수 없습니다: {image_path}")
                if index and not writer.newPage():
                    raise OSError(f"PDF 검사 {index + 1}페이지를 만들지 못했습니다.")
                painter.drawImage(target, image)
        finally:
            if not painter.end():
                raise OSError("가상 PDF 저장을 완료하지 못했습니다.")

    def print_selected(
        self, paths: list[Path] | None = None, scope_label: str = "선택한",
    ) -> None:
        paths = self.selected_paths() if paths is None else paths
        job_log_path = self._start_print_attempt(scope_label, paths)
        if not paths:
            self._finish_print_attempt("no_files")
            QMessageBox.warning(
                self, "인쇄할 사진이 없습니다", "인쇄할 사진이 목록에 없습니다."
            )
            return

        print_items: list[tuple[Path, CropResult | None, Path]] = []
        missing: list[Path] = []
        for source_path in paths:
            result = self.results.get(str(source_path))
            print_path = self._printable_path(source_path)
            if print_path:
                print_items.append((source_path, result, print_path))
            else:
                missing.append(source_path)
        if self._active_print_attempt:
            self._active_print_attempt["data"]["files"] = [
                {
                    "order": index + 1,
                    "source_path": str(source_path.resolve()),
                    "source_exists": source_path.is_file(),
                    "result_status": result.status if result else "not_processed",
                    "result_path": str(result.output_path) if result and result.output_path else None,
                    "print_path": str(print_path) if print_path else None,
                    "print_file_exists": bool(print_path and print_path.is_file()),
                    "missing_print_result": print_path is None,
                }
                for index, source_path in enumerate(paths)
                for result in [self.results.get(str(source_path))]
                for print_path in [self._printable_path(source_path)]
            ]
            self._record_print_event(
                "files_resolved", ready_count=len(print_items), missing_count=len(missing),
            )
        if missing:
            names = "\n".join(path.name for path in missing[:5])
            extra = f"\n외 {len(missing) - 5}개" if len(missing) > 5 else ""
            QMessageBox.warning(
                self, "보정 결과가 없는 사진",
                f"{scope_label} {len(paths)}개 중 {len(missing)}개에 인쇄할 결과가 없습니다.\n"
                f"먼저 해당 사진을 보정하세요.\n\n{names}{extra}",
            )
            self._finish_print_attempt(
                "missing_print_results",
                missing_files=[str(path.resolve()) for path in missing],
            )
            return

        available = QPrinterInfo.availablePrinterNames()
        self._record_print_event("printers_discovered", available_printers=available)
        if not available:
            QMessageBox.critical(
                self, "프린터를 찾을 수 없음",
                "운영체제에서 사용할 수 있는 프린터를 찾지 못했습니다. "
                "SELPHY 드라이버 설치와 프린터 연결을 확인하세요.",
            )
            self._finish_print_attempt("no_printer_available")
            return

        preferred_name = _preferred_printer_name()
        # Qt's native Windows print dialog reloads the driver's saved DEVMODE
        # and ignores the page layout seeded on QPrinter. It then shows
        # portrait even though the job is changed back to landscape below.
        # Use an app-owned confirmation dialog so the required C/landscape
        # settings are explicit and applied consistently on every print.
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{scope_label} {len(print_items)}개 사진 인쇄")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        form = QFormLayout(dialog)
        printer_combo = QComboBox(dialog)
        printer_combo.addItems(available)
        if preferred_name in available:
            printer_combo.setCurrentText(preferred_name)
        form.addRow("프린터", printer_combo)
        form.addRow("인쇄 매수", QLabel(f"{len(print_items)}장 · 사진마다 한 장씩"))
        form.addRow("용지", QLabel("C 카드 (54 × 86 mm)"))
        form.addRow("방향", QLabel("가로"))
        form.addRow(
            "인쇄 배치",
            QLabel("크롭 이미지 전체 출력 · 최대 81 × 51 mm", dialog),
        )
        self._record_print_event(
            "print_dialog_opened",
            candidate_printer=preferred_name,
            expected_pages=len(print_items),
            media="C Size",
            orientation="Landscape",
            layout="fit_complete_crop_inside_printable_area",
            target_size_mm=[81.0, 51.0],
        )
        preferences_button = QPushButton("프린터 드라이버 설정 저장…", dialog)
        preferences_button.setVisible(sys.platform == "win32")
        preferences_button.clicked.connect(
            lambda: self.save_printer_preferences(printer_combo.currentText(), dialog),
        )
        form.addRow(preferences_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        requested_action = {"value": "print"}
        pdf_button = buttons.addButton(
            "가상 PDF 검사", QDialogButtonBox.ButtonRole.ActionRole,
        )
        pdf_button.setToolTip(
            "종이를 쓰지 않고 Canon 프린터 기준 배치를 86 × 54mm PDF로 저장합니다."
        )
        pdf_button.clicked.connect(
            lambda: (requested_action.update(value="pdf"), dialog.accept())
        )
        print_button = buttons.addButton("인쇄", QDialogButtonBox.ButtonRole.AcceptRole)
        print_button.clicked.connect(lambda: requested_action.update(value="print"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._finish_print_attempt("cancelled_in_print_dialog")
            return
        virtual_pdf = requested_action["value"] == "pdf"

        selected_printer = printer_combo.currentText()
        layout_name = "fit_complete_crop_inside_printable_area"
        photo_size_mm = [81.0, 51.0]
        margins_mm = [2.5, 1.5]
        if self._active_print_attempt:
            self._active_print_attempt["data"].update(
                layout=layout_name,
                photo_size_mm=photo_size_mm,
                margins_mm=margins_mm,
            )
        self._record_print_event(
            "print_layout_selected",
            layout=layout_name,
            photo_size_mm=photo_size_mm,
            margins_mm=margins_mm,
        )
        settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        settings.setValue("printing/last_printer", selected_printer)
        settings.sync()
        self._record_print_event("printer_selected", printer_name=selected_printer)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPrinterName(selected_printer)
        _set_c_landscape_default(printer)
        printer.setDocName(f"신분증 사진 {len(print_items)}장")

        # Keep QPrinter in printable-area mode so pageRect() reports the area
        # the selected driver can actually mark. Full-page mode makes pageRect
        # equal paperRect and can let the image extend into an unprintable edge.
        # The whole crop is fitted inside the driver's area; it is never clipped.
        # The selected media/orientation are reapplied for every print job.
        printer.setFullPage(False)
        paper = printer.paperRect(QPrinter.Unit.DevicePixel)
        page = printer.pageRect(QPrinter.Unit.DevicePixel)
        paper_mm = printer.paperRect(QPrinter.Unit.Millimeter)
        paper_pt = printer.paperRect(QPrinter.Unit.Point)
        page_mm = printer.pageRect(QPrinter.Unit.Millimeter)
        printable = QRectF(page).intersected(QRectF(paper))
        is_c_landscape = (
            abs(paper_mm.width() - 86.0) <= 0.5
            and abs(paper_mm.height() - 54.0) <= 0.5
        )
        printer_info = QPrinterInfo(printer)
        page_layout = printer.pageLayout()
        driver_margins_mm = page_layout.margins(QPageLayout.Unit.Millimeter)
        supported_sizes = []
        for supported_size in printer_info.supportedPageSizes():
            size_mm = supported_size.size(QPageSize.Unit.Millimeter)
            supported_sizes.append({
                "name": supported_size.name(),
                "width_mm": size_mm.width(),
                "height_mm": size_mm.height(),
            })
        printer_geometry = {
            "printer_name": selected_printer,
            "printer_description": printer_info.description(),
            "printer_make_and_model": printer_info.makeAndModel(),
            "printer_is_default": printer_info.isDefault(),
            "printer_state_before_job": printer.printerState().name,
            # QPrinter exposes the selected mode at construction, but PySide6
            # does not provide a printerMode() getter.
            "printer_mode_requested": QPrinter.PrinterMode.HighResolution.name,
            "full_page": printer.fullPage(),
            "resolution_dpi": printer.resolution(),
            "logical_dpi": [printer.logicalDpiX(), printer.logicalDpiY()],
            "physical_dpi": [printer.physicalDpiX(), printer.physicalDpiY()],
            "paper_rect_mm": [paper_mm.x(), paper_mm.y(), paper_mm.width(), paper_mm.height()],
            "paper_rect_device_px": [paper.x(), paper.y(), paper.width(), paper.height()],
            "page_rect_device_px": [page.x(), page.y(), page.width(), page.height()],
            "paper_rect_points": [paper_pt.x(), paper_pt.y(), paper_pt.width(), paper_pt.height()],
            "page_rect_mm": [page_mm.x(), page_mm.y(), page_mm.width(), page_mm.height()],
            "printable_rect_device_px": self._rect_diagnostics(printable),
            "page_layout": {
                "orientation": page_layout.orientation().name,
                "mode": page_layout.mode().name,
                "page_size_name": page_layout.pageSize().name(),
                "margins_mm": {
                    "left": driver_margins_mm.left(),
                    "top": driver_margins_mm.top(),
                    "right": driver_margins_mm.right(),
                    "bottom": driver_margins_mm.bottom(),
                },
            },
            "driver_supported_page_sizes": supported_sizes,
            "runtime": {
                "platform": sys.platform,
                "python": sys.version,
                "qt_version": qVersion(),
            },
            "orientation_expected": "Landscape",
            "c_landscape_detected": is_c_landscape,
        }
        if self._active_print_attempt:
            self._active_print_attempt["data"]["printer"] = printer_geometry
        self._record_print_event("printer_geometry_read", **printer_geometry)
        if (not is_c_landscape or paper.width() <= 0 or paper.height() <= 0
                or printable.isEmpty()):
            for source_path, item_result, _ in print_items:
                self._write_print_log(item_result, {
                    "status": "wrong_paper_size_or_orientation",
                    "printer_name": printer.printerName(),
                    "reported_paper_rect_mm": [paper_mm.x(), paper_mm.y(),
                                                paper_mm.width(), paper_mm.height()],
                    "paper_rect_px": [paper.x(), paper.y(), paper.width(), paper.height()],
                    "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
                    "printer_resolution_dpi": printer.resolution(),
                    "source": str(source_path),
                    "batch_count": len(print_items),
                })
            QMessageBox.warning(
                self, "용지 방향 확인",
                "프린터가 86 × 54mm C 용지 가로 방향을 적용하지 않았습니다.\n"
                f"프린터가 보고한 크기: {paper_mm.width():.1f} × {paper_mm.height():.1f}mm\n"
                "SELPHY 드라이버에서 C 용지와 가로 방향을 선택하고 다시 인쇄하세요.",
            )
            self._finish_print_attempt("printer_media_mismatch")
            return

        loaded_items: list[tuple[Path, CropResult | None, Path]] = []
        for source_path, item_result, print_path in print_items:
            reader = QImageReader(str(print_path))
            if not reader.canRead():
                self._write_print_log(item_result, {
                    "status": "failed_to_load_image",
                    "printer_name": printer.printerName(),
                    "source": str(source_path),
                    "output_path": str(print_path),
                    "batch_count": len(print_items),
                    "error": reader.errorString(),
                })
                QMessageBox.critical(
                    self, "인쇄 오류", f"이미지를 열 수 없습니다.\n{print_path.name}"
                )
                self._finish_print_attempt(
                    "image_load_failed", failed_path=str(print_path),
                    image_reader_error=reader.errorString(),
                )
                return
            image_size = reader.size()
            self._record_print_event(
                "image_readable", path=str(print_path),
                width_px=image_size.width(), height_px=image_size.height(),
            )
            loaded_items.append((source_path, item_result, print_path))

        desired_width = paper.width() * 81.0 / paper_mm.width()
        desired_height = paper.height() * 51.0 / paper_mm.height()
        fit_scale = min(
            1.0,
            printable.width() / desired_width,
            printable.height() / desired_height,
        )
        target_width = desired_width * fit_scale
        target_height = desired_height * fit_scale
        target_x = QRectF(paper).center().x() - target_width / 2.0
        target_y = QRectF(paper).center().y() - target_height / 2.0
        target_x = min(max(target_x, printable.left()), printable.right() - target_width)
        target_y = min(max(target_y, printable.top()), printable.bottom() - target_height)
        target = QRectF(target_x, target_y, target_width, target_height)
        # QPrinter with fullPage=False uses pageRect's top-left as the painter
        # origin. Keep both coordinate interpretations in the diagnostic log.
        target_painter_origin = target.translated(-page.x(), -page.y())
        target_if_paper_coordinates_are_passed = target.translated(page.x(), page.y())
        paper_scale_x = paper_mm.width() / paper.width()
        paper_scale_y = paper_mm.height() / paper.height()
        photo_size_mm = [target_width * paper_scale_x, target_height * paper_scale_y]
        margins_mm = [
            (target.left() - paper.left()) * paper_scale_x,
            (target.top() - paper.top()) * paper_scale_y,
        ]
        if self._active_print_attempt:
            self._active_print_attempt["data"].update(
                photo_size_mm=photo_size_mm,
                margins_mm=margins_mm,
                fit_scale=fit_scale,
            )
        self._record_print_event(
            "print_target_fitted",
            target_rect_device_px=[target.x(), target.y(), target.width(), target.height()],
            target_rect_paper_coordinates=self._rect_diagnostics(target),
            target_rect_if_painter_origin_is_printable_area=self._rect_diagnostics(
                target_if_paper_coordinates_are_passed
            ),
            target_rect_translated_for_printable_origin=self._rect_diagnostics(
                target_painter_origin
            ),
            coordinate_origin_offset_device_px=[page.x(), page.y()],
            full_page_coordinate_origin=printer.fullPage(),
            actual_image_size_mm=photo_size_mm,
            fit_scale=fit_scale,
            printable_rect_device_px=[
                printable.x(), printable.y(), printable.width(), printable.height(),
            ],
        )
        if self._active_print_attempt:
            self._active_print_attempt["data"]["expected_page_count"] = len(loaded_items)
            self._active_print_attempt["data"]["target_rect_device_px"] = [
                target.x(), target.y(), target.width(), target.height(),
            ]

        if virtual_pdf:
            attempt_id = (
                self._active_print_attempt["data"]["attempt_id"]
                if self._active_print_attempt else "print_proof"
            )
            default_pdf = loaded_items[0][2].parent / f"{attempt_id}_virtual-proof.pdf"
            pdf_path, _ = QFileDialog.getSaveFileName(
                self, "가상 인쇄 PDF 저장", str(default_pdf), "PDF 파일 (*.pdf)",
            )
            if not pdf_path:
                self._finish_print_attempt("virtual_pdf_cancelled")
                return
            pdf_path = Path(pdf_path)
            if pdf_path.suffix.lower() != ".pdf":
                pdf_path = pdf_path.with_suffix(".pdf")
            try:
                self._export_virtual_print_pdf(
                    pdf_path, loaded_items, photo_size_mm, margins_mm,
                    printer.resolution(),
                )
            except Exception as exc:
                self._record_print_event(
                    "virtual_pdf_failed", error=f"{type(exc).__name__}: {exc}",
                )
                self._finish_print_attempt(
                    "virtual_pdf_failed", pdf_path=str(pdf_path),
                    error=f"{type(exc).__name__}: {exc}",
                    physical_print_confirmed=False,
                )
                QMessageBox.critical(
                    self, "가상 PDF 생성 실패",
                    f"PDF 검사 파일을 저장하지 못했습니다.\n{exc}",
                )
                return

            if self._active_print_attempt:
                self._active_print_attempt["data"]["pages"] = [
                    {
                        "page_number": index + 1,
                        "source_path": str(source_path.resolve()),
                        "image_path": str(image_path),
                        "new_page_called": index > 0,
                        "draw_called": True,
                        "virtual_pdf": True,
                    }
                    for index, (source_path, _, image_path) in enumerate(loaded_items)
                ]
            self._record_print_event(
                "virtual_pdf_saved",
                pdf_path=str(pdf_path),
                page_size_mm=[86.0, 54.0],
                photo_size_mm=photo_size_mm,
                margins_mm=margins_mm,
            )
            self._finish_print_attempt(
                "virtual_pdf_saved", pdf_path=str(pdf_path),
                expected_page_count=len(loaded_items),
                rendered_page_count=len(loaded_items),
                physical_print_confirmed=False,
            )
            QMessageBox.information(
                self, "가상 PDF 저장 완료",
                "종이를 사용하지 않고 가상 인쇄 PDF를 저장했습니다.\n\n"
                f"{pdf_path}\n\n"
                "PDF 안에 오른쪽 흰 선이 있으면 앱의 이미지 배치 쪽을, "
                "없으면 Canon 드라이버로 보내는 과정의 차이를 더 확인하면 됩니다.",
            )
            return

        painter = QPainter()
        if not painter.begin(printer):
            state = printer.printerState()
            for source_path, item_result, _ in loaded_items:
                self._write_print_log(item_result, {
                    "status": "painter_begin_failed",
                    "printer_name": printer.printerName(),
                    "printer_state": state.name,
                    "source": str(source_path),
                    "batch_count": len(loaded_items),
                })
            QMessageBox.critical(self, "인쇄 오류", "프린터 작업을 시작하지 못했습니다.")
            self._finish_print_attempt(
                "painter_begin_failed", printer_state=state.name,
            )
            return
        self._record_print_event("painter_started", expected_pages=len(loaded_items))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        transform = painter.deviceTransform()
        painter_context = {
            "coordinate_origin": "printable_area" if not printer.fullPage() else "paper",
            "device_size_px": [painter.device().width(), painter.device().height()],
            "device_logical_dpi": [
                painter.device().logicalDpiX(), painter.device().logicalDpiY(),
            ],
            "device_physical_dpi": [
                painter.device().physicalDpiX(), painter.device().physicalDpiY(),
            ],
            "viewport": self._rect_diagnostics(painter.viewport()),
            "window": self._rect_diagnostics(painter.window()),
            "device_transform": [
                transform.m11(), transform.m12(), transform.m21(), transform.m22(),
                transform.dx(), transform.dy(),
            ],
            "has_clip": painter.hasClipping(),
            "clip_rect_logical": self._rect_diagnostics(painter.clipBoundingRect()),
        }
        self._record_print_event("painter_context", **painter_context)
        rendered_count = 0
        page_error = False
        page_failure = None
        for index, (source_path, item_result, print_path) in enumerate(loaded_items):
            image = QImage(str(print_path))
            if image.isNull():
                page_error = True
                page_failure = {"page_number": index + 1, "reason": "image_load_failed",
                                "path": str(print_path)}
                break
            new_page_called = index > 0
            new_page_succeeded = not new_page_called or printer.newPage()
            if not new_page_succeeded:
                page_error = True
                page_failure = {"page_number": index + 1, "reason": "new_page_failed"}
                break
            transform = painter.deviceTransform()
            mapped_target = transform.mapRect(target)
            painter_clip = painter.clipBoundingRect()
            logical_intersection = target.intersected(painter_clip)
            preview_info = None
            preview_error = None
            try:
                attempt_log = self._active_print_attempt["path"]
                preview_path = attempt_log.with_name(
                    f"{attempt_log.stem}_page_{index + 1:02d}_app-render.png"
                )
                preview_info = self._write_print_geometry_preview(
                    preview_path, image, target, paper, page, printer.fullPage(),
                )
            except Exception as exc:
                # Diagnostics must never prevent or alter the requested print.
                preview_error = f"{type(exc).__name__}: {exc}"
            bitmap_diagnostics = self._print_image_diagnostics(image, print_path)
            draw_context = {
                "page_number": index + 1,
                "target_rect_passed_to_draw_image": self._rect_diagnostics(target),
                "target_rect_mapped_to_device": self._rect_diagnostics(mapped_target),
                "target_intersection_with_painter_clip": self._rect_diagnostics(
                    logical_intersection
                ),
                "target_clipped_by_painter_px": {
                    "left": max(0.0, logical_intersection.left() - target.left()),
                    "top": max(0.0, logical_intersection.top() - target.top()),
                    "right": max(0.0, target.right() - logical_intersection.right()),
                    "bottom": max(0.0, target.bottom() - logical_intersection.bottom()),
                },
                "painter_has_clip": painter.hasClipping(),
                "painter_clip_rect_logical": self._rect_diagnostics(painter_clip),
                "page_rect_device_px": self._rect_diagnostics(page),
                "paper_rect_device_px": self._rect_diagnostics(paper),
                "painter_device_transform": [
                    transform.m11(), transform.m12(), transform.m21(), transform.m22(),
                    transform.dx(), transform.dy(),
                ],
                "app_side_render_preview": preview_info,
                "app_side_render_preview_error": preview_error,
                "bitmap": bitmap_diagnostics,
            }
            self._record_print_event("draw_geometry_and_bitmap", **draw_context)
            painter.drawImage(target, image)
            rendered_count += 1
            page_details = {
                "page_number": index + 1,
                "source_path": str(source_path.resolve()),
                "image_path": str(print_path),
                "image_size_px": [image.width(), image.height()],
                "image_sha256": bitmap_diagnostics["sha256"],
                "target_rect_passed_to_draw_image": self._rect_diagnostics(target),
                "target_rect_mapped_to_device": self._rect_diagnostics(mapped_target),
                "app_side_render_preview": preview_info,
                "app_side_render_preview_error": preview_error,
                "new_page_called": new_page_called,
                "new_page_succeeded": new_page_succeeded,
                "draw_called": True,
            }
            if self._active_print_attempt:
                self._active_print_attempt["data"]["pages"].append(page_details)
            self._record_print_event("page_drawn", **page_details)
            del image
        ended = painter.end()
        state = printer.printerState()
        succeeded = (
            ended and not page_error and rendered_count == len(loaded_items)
            and state != QPrinter.PrinterState.Aborted
        )
        finished_log_path = self._finish_print_attempt(
            "sent_to_spooler" if succeeded else "print_job_failed",
            printer_name=selected_printer,
            printer_state_after_job=state.name,
            painter_end_succeeded=ended,
            expected_page_count=len(loaded_items),
            rendered_page_count=rendered_count,
            page_failure=page_failure,
            spooler_submission_confirmed=succeeded,
            physical_print_confirmed=False,
        )
        for index, (source_path, item_result, print_path) in enumerate(loaded_items):
            item_sent = succeeded and index < rendered_count
            try:
                self._write_print_log(item_result, {
                    "status": "sent_to_spooler" if item_sent else "print_job_failed",
                    "printer_name": printer.printerName(),
                    "printer_state": state.name,
                    "painter_end": ended,
                    "paper_rect_px": [paper.x(), paper.y(), paper.width(), paper.height()],
                    "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
                    "target_rect_px": [target.x(), target.y(), target.width(), target.height()],
                    "printer_resolution_dpi": printer.resolution(),
                    "page_size_mm": [86.0, 54.0],
                    "reported_paper_size_mm": [paper_mm.width(), paper_mm.height()],
                    "media": "C card (54 x 86 mm)",
                    "orientation": "landscape",
                    "image_target_mm": photo_size_mm,
                    "layout": layout_name,
                    "source": str(source_path),
                    "output_path": str(print_path),
                    "page_number": index + 1,
                    "batch_count": len(loaded_items),
                    "job_log": str(finished_log_path or job_log_path),
                })
            except OSError as exc:
                self._record_print_event(
                    "per_image_log_write_failed", source_path=str(source_path),
                    error=f"{type(exc).__name__}: {exc}",
                )
        if not succeeded:
            log_message = f"인쇄 로그: {finished_log_path or job_log_path}"
            if self.last_print_log_error:
                log_message += f"\n로그 저장 오류: {self.last_print_log_error}"
            QMessageBox.critical(
                self, "인쇄 오류",
                f"인쇄 작업 전송에 실패했습니다. 프린터 상태: {state.name}\n"
                f"{log_message}",
            )
            return
        if self.last_print_log_error:
            self.status.setText(
                f"{len(loaded_items)}개 사진을 인쇄 대기열로 보냈습니다. "
                f"로그 저장 실패: {self.last_print_log_error}"
            )
        else:
            self.status.setText(
                f"{len(loaded_items)}개 사진을 인쇄 대기열로 보냈습니다. "
                f"인쇄 로그: {finished_log_path or job_log_path}"
            )

    def handle_print_clicked(
        self, paths: list[Path] | None = None, scope_label: str = "선택한",
    ) -> None:
        """Make failures before the print dialog visible in packaged builds."""
        self.status.setText("인쇄 버튼이 눌렸습니다. 프린터와 출력 파일을 확인하고 있습니다…")
        QApplication.processEvents()
        try:
            self.print_selected(paths, scope_label)
        except Exception as exc:
            message = f"인쇄 창을 여는 중 오류가 발생했습니다: {type(exc).__name__}: {exc}"
            self.status.setText(message)
            self._finish_print_attempt(
                "exception", error_type=type(exc).__name__, error=str(exc),
            )
            QMessageBox.critical(self, "인쇄 창을 열 수 없음", message)

    def save_printer_preferences(self, printer_name: str, parent: QDialog) -> None:
        if not printer_name:
            return
        try:
            saved = _save_windows_printer_preferences(printer_name, int(parent.winId()))
        except OSError as exc:
            QMessageBox.warning(parent, "프린터 설정 저장 실패", str(exc))
            return
        if not saved:
            return
        settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        settings.setValue("printing/last_printer", printer_name)
        settings.sync()
        QMessageBox.information(
            parent, "프린터 설정 저장 완료",
            "선택한 SELPHY 드라이버 설정을 현재 Windows 사용자 기본값으로 저장했습니다. "
            "다음 인쇄부터 이 설정이 적용됩니다.",
        )

    def _write_print_log(self, result: CropResult | None, info: dict) -> None:
        default_folder = Path(self.output_edit.text()).expanduser()
        folder = (
            result.debug_dir or (result.output_path.parent if result.output_path else default_folder)
            if result else default_folder
        )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "print_attempt.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def open_output(self) -> None:
        folder = Path(self.output_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def open_print_logs(self) -> None:
        folder = Path(self.output_edit.text()).expanduser() / "print_logs"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    @staticmethod
    def _find_photoshop_executable() -> Path | None:
        configured = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION).value(
            "editing/photoshop_executable", "",
        )
        if configured and Path(str(configured)).is_file():
            return Path(str(configured))

        if sys.platform == "win32":
            import winreg

            registry_paths = (
                (winreg.HKEY_CURRENT_USER,
                 r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"),
            )
            for root, key_path in registry_paths:
                try:
                    with winreg.OpenKey(root, key_path) as key:
                        executable = Path(winreg.QueryValueEx(key, None)[0])
                    if executable.is_file():
                        return executable
                except (OSError, ValueError):
                    continue
            for variable in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
                base = os.environ.get(variable)
                if not base:
                    continue
                matches = sorted(
                    Path(base).glob("Adobe/Adobe Photoshop*/Photoshop.exe"),
                    reverse=True,
                )
                if matches:
                    return matches[0]
        elif sys.platform == "darwin":
            apps = sorted(
                Path("/Applications").glob(
                    "Adobe Photoshop*.app/Contents/MacOS/Adobe Photoshop",
                ),
                reverse=True,
            )
            if apps:
                return apps[0]
        return None

    def edit_selected_images(self) -> None:
        sources = self.selected_paths()
        if not sources:
            QMessageBox.information(self, "사진을 선택하세요", "수정할 사진을 체크하세요.")
            return
        image_paths = [self._printable_path(source) or source for source in sources]
        image_paths = [path for path in image_paths if path.is_file()]
        if not image_paths:
            QMessageBox.warning(self, "열 사진이 없습니다", "선택한 사진 파일을 찾을 수 없습니다.")
            return
        executable = self._find_photoshop_executable()
        if executable is None and sys.platform == "win32":
            selected, _ = QFileDialog.getOpenFileName(
                self, "Photoshop 실행 파일 선택", "",
                "Photoshop (Photoshop.exe);;실행 파일 (*.exe)",
            )
            if not selected:
                return
            executable = Path(selected)
            QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION).setValue(
                "editing/photoshop_executable", str(executable),
            )
            QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION).sync()
        if executable is None:
            QMessageBox.warning(
                self, "Photoshop을 찾을 수 없음",
                "Photoshop 설치 경로를 찾지 못했습니다. Photoshop을 설치했는지 확인하세요.",
            )
            return
        self.pending_photoshop_paths = image_paths
        self.photoshop_executable = executable
        self.photoshop_launch_pending = True
        self.update_selection_controls()
        self.status.setText(f"Photoshop에서 {len(image_paths)}개 사진을 순서대로 여는 중입니다…")
        self._launch_next_photoshop_image()

    def _launch_next_photoshop_image(self) -> None:
        if not self.pending_photoshop_paths:
            self.photoshop_launch_pending = False
            self.update_selection_controls()
            self.status.setText("선택한 사진을 모두 Photoshop에 열었습니다.")
            return
        image_path = self.pending_photoshop_paths.pop(0)
        try:
            if sys.platform == "darwin":
                app_path = self.photoshop_executable.parent.parent.parent
                subprocess.Popen(["open", "-a", str(app_path), str(image_path)])
            else:
                subprocess.Popen([str(self.photoshop_executable), str(image_path)])
        except OSError as exc:
            self.pending_photoshop_paths.clear()
            self.photoshop_launch_pending = False
            self.update_selection_controls()
            QMessageBox.critical(
                self, "Photoshop 실행 실패", f"이미지를 Photoshop에서 열지 못했습니다.\n{exc}",
            )
            return
        if self.pending_photoshop_paths:
            QTimer.singleShot(700, self._launch_next_photoshop_image)
        else:
            self.photoshop_launch_pending = False
            self.update_selection_controls()
            self.status.setText("선택한 사진을 모두 Photoshop에 열었습니다.")

    def set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        for button in (self.add_files_button, self.add_folder_button, self.remove_button,
                       self.remove_all_button,
                       self.process_selected_button, self.process_all_button,
                       self.clear_selection_button,
                       self.print_button, self.print_all_button, self.edit_selected_button,
                       self.recognition_enhance_button, self.undo_button,
                       self.open_print_logs_button,
                       self.output_browse_button):
            button.setEnabled(not busy)
        self.output_edit.setEnabled(not busy)
        self.file_list.setEnabled(not busy)
        self.select_all_checkbox.setEnabled(not busy and self.file_list.count() > 0)
        if not busy:
            path = self.selected_path()
            self.recognition_enhance_button.setEnabled(bool(path and path.is_file()))
            self.update_selection_controls()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("셀피 카드 스튜디오")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
