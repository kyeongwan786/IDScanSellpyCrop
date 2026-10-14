# 신분증 스캔 자동 보정

셀피 카드 스튜디오는 신분증 사진을 로컬에서 기울기 보정·크롭하고 인쇄하는 GUI 프로그램입니다. 원본 사진의 전체 기울기를 먼저 추정해 바로 세운 뒤 카드 외곽을 검출하고, 원근을 펴서 경계보다 약간 안쪽을 크롭합니다. 결과 JPEG의 문서 크기는 8.1 × 5.1cm입니다.

파일 선택, 미리보기, 일괄 보정, SELPHY 등 연결 프린터로 인쇄하는 GUI 실행 파일도 제공합니다.

## 설치

Python 3.10 이상에서 프로젝트 폴더를 연 뒤:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install '.[ocr,gui]'
```

설치 후 `idcard-crop-gui`를 실행하면 파일 선택 화면이 열립니다. 소스에서 직접 실행할 때는 프로젝트 폴더에서 `python run_gui.py`를 사용합니다.

방향 판별에는 Tesseract OCR 실행 파일과 OSD 방향 판별 모델이 필요합니다. Windows EXE 빌드 방법은 아래 항목을 참고하세요. 방향 판별은 로컬에서 처리합니다.

## Windows EXE 빌드

Windows x64용 Python 3.12와 Tesseract OCR을 설치한 뒤 프로젝트 폴더에서 `build_windows.bat`을 실행합니다. 스크립트는 `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`를 먼저 검사하고 AMD64 Python을 선택합니다. 기존 `.venv-win`이 AMD64가 아니면 다시 만들며, 최종 EXE의 Windows x64 아키텍처도 검사합니다. 기본 Tesseract 설치 경로는 `%ProgramFiles%\Tesseract-OCR`입니다. 다른 경로에 설치했다면 `TESSERACT_DIR` 환경 변수에 설치 폴더를 지정합니다. GUI, Tesseract 실행 파일, 한국어 OCR 모델을 포함한 `dist\IDCardCropper.exe`를 생성합니다.

```bat
dist\IDCardCropper.exe
```

체크한 사진만 처리할 때는 **선택 항목 보정**, 목록의 모든 사진을 처리할 때는 **전체 보정**을 누릅니다. 체크박스로 인쇄하거나 수정할 사진을 고른 뒤 **선택 삭제**와 **전체 삭제**를 쓸 수 있으며, 삭제는 앱 목록에서만 이뤄지고 원본 파일은 지우지 않습니다. **선택 항목 인쇄**는 체크된 사진을, **전체 인쇄**는 목록에 있는 모든 사진을 목록 순서대로 인쇄합니다. 여러 장은 사진마다 한 페이지씩 한 인쇄 작업으로 전송됩니다. 인쇄 전 모든 대상 사진의 보정이 끝나 있어야 합니다. **수정하기**는 체크된 사진을 순서대로 Photoshop에서 엽니다. Photoshop 설정에서 문서를 탭이 아닌 별도 창으로 열도록 해두면 각각의 사진 창으로 볼 수 있습니다. Windows에서 Photoshop을 자동으로 찾지 못하면 처음 한 번 `Photoshop.exe` 위치를 고릅니다. Photoshop 창이 어느 모니터에 표시되는지는 Photoshop과 Windows의 창 배치에 따릅니다.

사진을 추가할 때 체크는 해제된 상태입니다. 사진 목록의 체크박스뿐 아니라 행의 빈 곳을 클릭해도 선택 상태가 바뀝니다. 보정 후에는 미리보기가 **보정 결과** 탭으로 전환되고, 마우스 휠로 미리보기 확대·축소가 가능합니다. **인식 보정**을 되돌리려면 **되돌리기** 또는 `Ctrl+Z`를 누르세요.

인쇄를 누를 때마다 별도 JSON 로그가 결과 폴더의 `print_logs` 안에 저장됩니다. 선택 파일과 출력 파일, 프린터와 용지 정보, 페이지별 전송 여부, 스풀러 전달 상태를 확인할 수 있습니다. GUI의 **인쇄 로그 열기** 버튼으로 폴더를 열 수 있습니다. 로그의 `sent_to_spooler`는 Windows 인쇄 대기열로 전달됐다는 뜻이며, 프린터에서 실제 인쇄가 완료됐는지를 확인하는 값은 아닙니다.

Windows에서는 인쇄 창의 **프린터 드라이버 설정 저장…**을 눌러 SELPHY 드라이버의 용지와 무테 등 옵션을 설정할 수 있습니다. 저장한 설정은 현재 Windows 사용자와 해당 프린터에 적용됩니다. 앱은 크롭 JPG 전체를 절취 탭 안쪽의 C 카드 영역(86 × 54mm)에 맞춰 배치합니다. 드라이버의 출력 가능 영역 대신 C 카드 영역을 사용하므로, 이미지가 카드 영역을 채우고 크롭 결과의 가장자리도 앱에서 잘라내지 않습니다.

출력 폴더에는 진단 이미지와 정상 처리된 인화용 JPG가 생성됩니다. 방향을 확인할 수 없는 파일은 진단 이미지만 만들고 수동 확인이 필요한 상태로 남깁니다. 원본은 수정하지 않습니다.

각 이미지 처리 단계는 결과 폴더의 `<파일명>_debug` 폴더에 저장됩니다. 전체 사진 기울기를 먼저 보정한 경우 `00_source_deskewed.jpg`와 추정 각도가 기록됩니다. 카드 경계와 최종 크롭은 `01_card_boundary_and_crop.jpg`, 결과 이미지는 `02_final.jpg`에서 확인할 수 있습니다. 인쇄를 시도하면 프린터·용지·인쇄 전송 정보도 `print_attempt.json`에 기록됩니다. 진단 이미지에는 신분증 정보가 포함되므로 외부 공유 전에 삭제하세요.

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
