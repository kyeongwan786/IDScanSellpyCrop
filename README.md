# 신분증 스캔 자동 보정

Python/OpenCV 기반 1차 로컬 처리 엔진입니다. 이미지 검출, 사각형 원근 보정, OCR 방향 판별(선택 의존성), 8.1 × 5.1cm / 300 DPI JPG 출력을 제공합니다. 인화 크기에 맞추기 위해 가로세로 비율을 최종 단계에서 미세 조정하며, 자동 검출이나 방향 판별이 불확실하면 최종 이미지를 저장하지 않고 `manual_review`로 반환합니다.

파일 선택, 미리보기, 일괄 보정, SELPHY 등 연결 프린터로 인쇄하는 GUI 실행 파일도 제공합니다.

## 설치

Python 3.10 이상에서 프로젝트 폴더를 연 뒤:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install '.[ocr]'
```

GUI까지 설치하려면 `python -m pip install '.[ocr,gui]'`를 사용하세요. 설치 후 `idcard-crop-gui`를 실행하면 파일 선택 화면이 열립니다. 소스에서 직접 실행할 때는 프로젝트 폴더에서 `python run_gui.py`를 사용합니다.

방향 판별에는 Python 패키지 외에 Tesseract OCR 실행 파일이 필요합니다. 한국어 OCR 모델은 프로젝트에 포함되어 네 방향의 인식 신뢰도만 로컬 비교합니다. 인식한 문장은 저장하거나 출력하지 않습니다. OCR을 설치하지 않았거나 판별이 불확실하면 자동 저장 대신 수동 확인 상태가 됩니다. PDF 지원은 다음 단계에서 추가할 예정입니다.

## 사용법

```bash
idcard-crop /path/to/scan.jpg -o output
idcard-crop /path/to/scans/ -o output
```

## Windows EXE 빌드

Windows x64용 Python 3.12와 Tesseract OCR을 설치한 뒤 프로젝트 폴더에서 `build_windows.bat`을 실행합니다. 스크립트는 `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`를 먼저 검사하고 AMD64 Python을 선택합니다. 기존 `.venv-win`이 AMD64가 아니면 다시 만들며, 최종 EXE의 Windows x64 아키텍처도 검사합니다. 기본 Tesseract 설치 경로는 `%ProgramFiles%\Tesseract-OCR`입니다. 다른 경로에 설치했다면 `TESSERACT_DIR` 환경 변수에 설치 폴더를 지정합니다. GUI, Tesseract 실행 파일, 한국어 OCR 모델을 포함한 `dist\IDCardCropper.exe`를 생성합니다.

```bat
dist\IDCardCropper.exe
```

GUI에서 **이미지 추가** 또는 **폴더 추가**를 선택하고, 결과를 확인한 뒤 **보정 결과 인쇄**를 누릅니다. 인쇄 대화상자에서 연결된 SELPHY와 카드 용지를 선택합니다. 앱은 8.1 × 5.1cm 이미지를 선택한 용지의 가운데 실제 크기로 배치합니다. SELPHY 모델과 드라이버에 따라 카드 용지/테두리 없는 인쇄 설정 이름이 다를 수 있습니다.

출력 폴더에는 진단 이미지와 정상 처리된 인화용 JPG가 생성됩니다. 방향을 확인할 수 없는 파일은 진단 이미지만 만들고 수동 확인이 필요한 상태로 남깁니다. 원본은 수정하지 않습니다.

각 이미지 처리 단계는 결과 폴더의 `<파일명>_debug` 폴더에 저장됩니다. `단계별 진단 열기`를 누르면 바로 열 수 있습니다. 폴더에는 원본, 바깥 후보 점, 안쪽 크롭 점, 검출 마스크, 기울기 보정 결과, 원근 보정 결과, 방향 보정 결과, 인쇄 비율에 맞춘 중앙 크롭, 최종 출력 이미지와 `run.json` 로그가 포함됩니다. 로그에는 점 좌표와 검출 점수, 회전 각도, 각 단계의 크기가 기록되며 OCR로 인식한 글자는 기록하지 않습니다. 진단 이미지는 신분증 정보를 포함하므로 해당 PC의 출력 폴더에만 보관하세요.

이미지 처리는 외부 서비스에 전송하지 않고 로컬에서 수행합니다. 개인 정보가 담긴 스캔본은 저장소에 추가하지 마세요.

## GitHub에서 Windows EXE 자동 빌드

`.github/workflows/windows-build.yml`은 `main` 또는 `master`에 push할 때마다 Windows x64 환경에서 EXE를 빌드합니다. 빌드가 끝나면 GitHub 저장소의 **Actions**에서 `IDCardCropper-windows-x64` 파일을 받을 수 있습니다. 이 자동 빌드 파일은 30일간 보관됩니다.

테스트 PC에 전달할 버전 릴리즈를 만들려면 Mac에서 버전 태그를 push합니다. 예를 들면:

```bash
git tag v0.1.0
git push origin v0.1.0
```

완료되면 저장소의 **Releases**에 `IDCardCropper.exe`가 올라옵니다. 테스트 PC에서는 Releases에서 최신 EXE 하나만 내려받아 실행하면 됩니다. 소스 코드를 push할 때마다 테스트 PC의 파일이 자동 교체되지는 않습니다. 새 EXE가 올라오면 테스트 PC에서 내려받아야 합니다.

저장소를 만들 때는 **Private**로 설정하고, 신분증 스캔 이미지나 출력 결과를 Git에 추가하지 마세요. 이 프로젝트의 `.gitignore`는 일반 이미지와 PDF 파일을 제외합니다.
