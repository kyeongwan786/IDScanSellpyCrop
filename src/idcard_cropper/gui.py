"""Desktop GUI for selecting, correcting, and printing ID card scans."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPageLayout, QPainter, QPixmap
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
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

from .engine import CropResult, process_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
USER_ROLE = int(Qt.ItemDataRole.UserRole)


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
        self.source_preview = Preview("원본 미리보기")
        self.result_preview = Preview("보정 결과 미리보기")
        preview_layout.addWidget(self._preview_column("원본", self.source_preview), 1)
        preview_layout.addWidget(self._preview_column("보정 결과", self.result_preview), 1)
        splitter.addWidget(previews)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.process_selected_button = QPushButton("선택 항목 보정")
        self.process_all_button = QPushButton("전체 보정")
        self.print_button = QPushButton("보정 결과 인쇄")
        self.open_output_button = QPushButton("결과 폴더 열기")
        actions.addWidget(self.process_selected_button)
        actions.addWidget(self.process_all_button)
        actions.addStretch(1)
        actions.addWidget(self.print_button)
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
        self.file_list.currentItemChanged.connect(self.selection_changed)
        self.print_button.clicked.connect(self.print_selected)
        self.open_output_button.clicked.connect(self.open_output)
        self.print_button.setEnabled(False)

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

    def start_processing(self, paths: list[Path]) -> None:
        output_dir = Path(self.output_edit.text()).expanduser()
        self.set_busy(True)
        self.status.setText(f"{len(paths)}개 이미지를 처리하고 있습니다…")
        self.worker = ProcessWorker(paths, output_dir)
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
            self.print_button.setEnabled(False)

    def display_path(self, path: Path) -> None:
        self.source_preview.show_image(path)
        result = self.results.get(str(path))
        if result:
            self.result_preview.show_image(result.output_path or result.diagnostic_path)
            self.print_button.setEnabled(bool(result.output_path and result.output_path.is_file()))
        else:
            self.result_preview.show_image(None)
            self.print_button.setEnabled(False)

    def print_selected(self) -> None:
        path = self.selected_path()
        result = self.results.get(str(path)) if path else None
        if not result or not result.output_path or not result.output_path.is_file():
            QMessageBox.warning(self, "인쇄할 결과 없음", "먼저 이미지를 보정하세요.")
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageOrientation(QPageLayout.Orientation.Landscape)
        printer.setDocName(result.output_path.stem)
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle("SELPHY 프린터와 용지 설정")
        if dialog.exec() != QPrintDialog.DialogCode.Accepted:
            return

        # Print the 81 x 51 mm card at its physical size, centered on the
        # printer's selected paper. Users choose SELPHY/card media in its driver.
        printer.setFullPage(True)
        page = printer.pageRect(QPrinter.Unit.DevicePixel)
        px_per_mm = printer.resolution() / 25.4
        card_w = 81.0 * px_per_mm
        card_h = 51.0 * px_per_mm
        if page.width() < card_w or page.height() < card_h:
            QMessageBox.warning(
                self, "용지 크기 확인",
                "선택한 용지가 8.1 × 5.1cm 인쇄 영역보다 작습니다.\n"
                "프린터 설정에서 SELPHY 카드 크기 또는 더 큰 용지를 선택하세요.",
            )
            return

        image = QImage(str(result.output_path))
        if image.isNull():
            QMessageBox.critical(self, "인쇄 오류", "보정 이미지를 열 수 없습니다.")
            return
        target = QRectF(page.x() + (page.width() - card_w) / 2,
                        page.y() + (page.height() - card_h) / 2,
                        card_w, card_h)
        painter = QPainter()
        if not painter.begin(printer):
            QMessageBox.critical(self, "인쇄 오류", "프린터 작업을 시작하지 못했습니다.")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, image)
        painter.end()
        self.status.setText("인쇄 작업을 프린터로 보냈습니다. 프린터 드라이버에서 카드 용지와 테두리 설정을 확인하세요.")

    def open_output(self) -> None:
        folder = Path(self.output_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def set_busy(self, busy: bool) -> None:
        for button in (self.add_files_button, self.add_folder_button, self.remove_button,
                       self.process_selected_button, self.process_all_button):
            button.setEnabled(not busy)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("신분증 스캔 보정 및 인쇄")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
