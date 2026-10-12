"""Desktop GUI for selecting, correcting, and printing ID card scans."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
from PySide6.QtCore import (Qt, QThread, Signal, QRectF, QUrl, QSizeF, QSettings, QEvent,
                            QPoint, QMimeData)
from PySide6.QtGui import (QDesktopServices, QDrag, QImage, QPageLayout, QPageSize, QPainter,
                           QPixmap)
from PySide6.QtPrintSupport import QPrinter, QPrinterInfo
from PySide6.QtWidgets import (
    QApplication,
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

from .engine import CropResult, enhance_for_recognition, process_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
USER_ROLE = int(Qt.ItemDataRole.UserRole)
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


class Preview(QLabel):
    def __init__(self, title: str, *, allow_file_drag: bool = False) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #20242b; color: #d7dbe2; border: 1px solid #444b55; }")
        self._pixmap: QPixmap | None = None
        self._file_path: Path | None = None
        self._allow_file_drag = allow_file_drag
        self._drag_origin: QPoint | None = None
        self._placeholder = title
        if allow_file_drag:
            self.setToolTip("이미지를 다른 앱으로 드래그하거나 더블클릭해 기본 사진 앱에서 여세요.")

    def show_image(self, path: Path | None) -> None:
        self._pixmap = QPixmap(str(path)) if path and path.is_file() else None
        self._file_path = Path(path).resolve() if self._pixmap and self._allow_file_drag else None
        self._render()

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._file_path = None
        self._render()

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
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
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
            result = process_image(path, self.output_dir)
            self.item_done.emit(str(path), result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("신분증 스캔 보정 및 인쇄")
        self.resize(1120, 720)
        self.setAcceptDrops(True)
        self.results: dict[str, CropResult] = {}
        self.recognition_outputs: dict[str, Path] = {}
        self.worker: ProcessWorker | None = None
        self.pending_processing: list[Path] = []

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
        self.source_preview = Preview("원본 미리보기")
        self.result_preview = Preview("보정 결과 미리보기", allow_file_drag=True)
        preview_layout.addWidget(self._preview_column("원본", self.source_preview), 1)
        preview_layout.addWidget(self._preview_column("보정 결과", self.result_preview), 1)
        splitter.addWidget(previews)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.crop_help = QLabel(
            "이미지를 추가해도 자동 처리하지 않습니다. 선택 항목 보정/전체 보정은 신분증을 크롭합니다. "
            "'인식 안 될 때 누르는 버튼'은 크롭 결과가 있으면 그 결과에, 없으면 원본에 색·밝기 보정만 추가 적용합니다. "
            "이 버튼은 새 크롭을 하지 않습니다. "
            "보정 결과는 결과 영역에서 다른 앱으로 드래그하거나 "
            "더블클릭해 기본 사진 앱에서 열 수 있습니다."
        )
        self.crop_help.setWordWrap(True)
        layout.addWidget(self.crop_help)

        actions = QHBoxLayout()
        self.process_selected_button = QPushButton("선택 항목 보정")
        self.process_all_button = QPushButton("전체 보정")
        self.recognition_enhance_button = QPushButton("인식 안 될 때 누르는 버튼")
        self.recognition_enhance_button.setToolTip(
            "비포·애프터 예시에서 추출한 밝기·채도 분포를 참고해 보정합니다. "
            "크롭 결과가 있으면 그 이미지에, 없으면 업로드 원본에 적용합니다."
        )
        self.print_button = QPushButton("보정 결과 인쇄")
        self.open_diagnostics_button = QPushButton("단계별 진단 열기")
        self.open_output_button = QPushButton("결과 폴더 열기")
        actions.addWidget(self.process_selected_button)
        actions.addWidget(self.process_all_button)
        actions.addWidget(self.recognition_enhance_button)
        actions.addStretch(1)
        actions.addWidget(self.print_button)
        actions.addWidget(self.open_diagnostics_button)
        actions.addWidget(self.open_output_button)
        layout.addLayout(actions)

        self.status = QLabel(
            "이미지를 추가했습니다. 선택 항목 보정/전체 보정 또는 인식 보정 버튼을 눌러 작업을 선택하세요."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.add_files_button.clicked.connect(self.add_files)
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button.clicked.connect(self.remove_selected)
        self.output_browse_button.clicked.connect(self.choose_output)
        self.process_selected_button.clicked.connect(self.process_selected)
        self.process_all_button.clicked.connect(self.process_all)
        self.recognition_enhance_button.clicked.connect(self.enhance_selected_for_recognition)
        self.file_list.currentItemChanged.connect(self.selection_changed)
        self.print_button.clicked.connect(self.handle_print_clicked)
        self.open_diagnostics_button.clicked.connect(self.open_diagnostics)
        self.open_output_button.clicked.connect(self.open_output)
        # Keep this actionable even before processing so a click can explain
        # why there is no printable result instead of appearing unresponsive.
        self.print_button.setEnabled(True)
        self.open_diagnostics_button.setEnabled(False)
        self.recognition_enhance_button.setEnabled(False)

        # Qt delivers drag events to the child widget under the pointer. Install
        # the same file-drop handler on the full UI so buttons, previews, and
        # the list all accept files just like the unused background does.
        for widget in [root, *root.findChildren(QWidget)]:
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)

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
        added_paths: list[Path] = []
        for path in paths:
            resolved = str(path.resolve())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and resolved not in known:
                item = QListWidgetItem(f"대기 중 · {path.name}")
                item.setData(USER_ROLE, resolved)
                self.file_list.addItem(item)
                known.add(resolved)
                added_paths.append(Path(resolved))
        if added_paths:
            self.file_list.setCurrentRow(self.file_list.count() - len(added_paths))
            self.status.setText(
                f"이미지 {len(added_paths)}개를 추가했습니다. 실행할 작업을 선택하세요."
            )

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
        row = self.file_list.currentRow()
        if row >= 0:
            item = self.file_list.takeItem(row)
            raw_path = item.data(USER_ROLE)
            self.results.pop(raw_path, None)
            self.recognition_outputs.pop(raw_path, None)
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

    def start_processing(self, paths: list[Path]) -> None:
        if self.worker and self.worker.isRunning():
            self.pending_processing.extend(paths)
            return
        for path in paths:
            self.recognition_outputs.pop(str(path), None)
        output_dir = Path(self.output_edit.text()).expanduser()
        self.set_busy(True)
        self.status.setText(
            f"{len(paths)}개 이미지의 기본 보정을 처리하고 있습니다…"
        )
        self.worker = ProcessWorker(paths, output_dir)
        self.worker.item_done.connect(self.processing_done)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def processing_finished(self) -> None:
        self.set_busy(False)
        if self.pending_processing:
            queued = self.pending_processing
            self.pending_processing = []
            self.start_processing(queued)

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
            self.source_preview.show_image(None)
            self.result_preview.show_image(None)
            self.open_diagnostics_button.setEnabled(False)
            self.recognition_enhance_button.setEnabled(False)

    def display_path(self, path: Path) -> None:
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
        self.recognition_enhance_button.setEnabled(path.is_file())

    @staticmethod
    def _recognition_output_path(input_path: Path, output_dir: Path) -> Path:
        return output_dir / f"{input_path.stem}_인식보정.jpg"

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
            self.recognition_outputs[str(source_path)] = enhanced_path
            details_path = output_dir / f"{source_path.stem}_인식보정.json"
            details_path.write_text(
                json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8",
            )
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
        recognition_path = self.recognition_outputs.get(str(path)) if path else None
        print_path = (recognition_path if recognition_path and recognition_path.is_file()
                      else result.output_path if result else None)
        if not print_path or not print_path.is_file():
            QMessageBox.warning(
                self, "인쇄할 결과 없음",
                "먼저 선택 항목 보정/전체 보정 또는 인식 보정 버튼을 눌러 결과를 만드세요.",
            )
            return

        available = QPrinterInfo.availablePrinterNames()
        if not available:
            QMessageBox.critical(
                self, "프린터를 찾을 수 없음",
                "운영체제에서 사용할 수 있는 프린터를 찾지 못했습니다. "
                "SELPHY 드라이버 설치와 프린터 연결을 확인하세요.",
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
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        form = QFormLayout(dialog)
        printer_combo = QComboBox(dialog)
        printer_combo.addItems(available)
        if preferred_name in available:
            printer_combo.setCurrentText(preferred_name)
        form.addRow("프린터", printer_combo)
        form.addRow("용지", QLabel("C 카드 (54 × 86 mm)"))
        form.addRow("방향", QLabel("가로"))
        form.addRow("인쇄 영역", QLabel("C 용지 전체 채움 (86 × 54 mm)"))
        preferences_button = QPushButton("프린터 드라이버 설정 저장…", dialog)
        preferences_button.setVisible(sys.platform == "win32")
        preferences_button.clicked.connect(
            lambda: self.save_printer_preferences(printer_combo.currentText(), dialog),
        )
        form.addRow(preferences_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        buttons.addButton("인쇄", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_printer = printer_combo.currentText()
        settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        settings.setValue("printing/last_printer", selected_printer)
        settings.sync()

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPrinterName(selected_printer)
        _set_c_landscape_default(printer)
        printer.setDocName(print_path.stem)

        # Fill the complete 86 x 54 mm C-size sheet. The saved JPG is 81 x 51 mm;
        # it is scaled to the paper edges here so the print has no app-added
        # white border. The selected media/orientation are reapplied each job.
        printer.setFullPage(True)
        paper = printer.paperRect(QPrinter.Unit.DevicePixel)
        page = printer.pageRect(QPrinter.Unit.DevicePixel)
        paper_mm = printer.paperRect(QPrinter.Unit.Millimeter)
        is_c_landscape = (
            abs(paper_mm.width() - 86.0) <= 0.5
            and abs(paper_mm.height() - 54.0) <= 0.5
        )
        if not is_c_landscape or paper.width() <= 0 or paper.height() <= 0:
            self._write_print_log(result, {
                "status": "wrong_paper_size_or_orientation",
                "printer_name": printer.printerName(),
                "reported_paper_rect_mm": [paper_mm.x(), paper_mm.y(),
                                            paper_mm.width(), paper_mm.height()],
                "paper_rect_px": [paper.x(), paper.y(), paper.width(), paper.height()],
                "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
                "printer_resolution_dpi": printer.resolution(),
            })
            QMessageBox.warning(
                self, "용지 방향 확인",
                "프린터가 86 × 54mm C 용지 가로 방향을 적용하지 않았습니다.\n"
                f"프린터가 보고한 크기: {paper_mm.width():.1f} × {paper_mm.height():.1f}mm\n"
                "SELPHY 드라이버에서 C 용지와 가로 방향을 선택하고 다시 인쇄하세요.",
            )
            return

        image = QImage(str(print_path))
        if image.isNull():
            self._write_print_log(result, {
                "status": "failed_to_load_image",
                "printer_name": printer.printerName(),
            })
            QMessageBox.critical(self, "인쇄 오류", "보정 이미지를 열 수 없습니다.")
            return
        target = QRectF(paper)
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
            "paper_rect_px": [paper.x(), paper.y(), paper.width(), paper.height()],
            "page_rect_px": [page.x(), page.y(), page.width(), page.height()],
            "target_rect_px": [target.x(), target.y(), target.width(), target.height()],
            "printer_resolution_dpi": printer.resolution(),
            "page_size_mm": [86.0, 54.0],
            "reported_paper_size_mm": [paper_mm.width(), paper_mm.height()],
            "media": "C card (54 x 86 mm)",
            "orientation": "landscape",
            "image_target_mm": [86.0, 54.0],
            "layout": "fill_entire_paper",
        })
        if not succeeded:
            QMessageBox.critical(
                self, "인쇄 오류",
                f"인쇄 작업 전송에 실패했습니다. 프린터 상태: {state.name}\n"
                "단계별 진단 폴더의 print_attempt.json을 확인하세요.",
            )
            return
        self.status.setText("인쇄 작업을 프린터로 보냈습니다. 프린터 드라이버에서 카드 용지와 테두리 설정을 확인하세요.")

    def handle_print_clicked(self) -> None:
        """Make failures before the print dialog visible in packaged builds."""
        self.status.setText("인쇄 버튼이 눌렸습니다. 프린터와 출력 파일을 확인하고 있습니다…")
        QApplication.processEvents()
        try:
            self.print_selected()
        except Exception as exc:
            message = f"인쇄 창을 여는 중 오류가 발생했습니다: {type(exc).__name__}: {exc}"
            self.status.setText(message)
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

    def set_busy(self, busy: bool) -> None:
        for button in (self.add_files_button, self.add_folder_button, self.remove_button,
                       self.process_selected_button, self.process_all_button, self.print_button,
                       self.recognition_enhance_button):
            button.setEnabled(not busy)
        if not busy:
            path = self.selected_path()
            self.recognition_enhance_button.setEnabled(bool(path and path.is_file()))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("신분증 스캔 보정 및 인쇄")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
