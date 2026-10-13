`kor.traineddata` is from the official Tesseract OCR `tessdata_fast` repository:
https://github.com/tesseract-ocr/tessdata_fast

It is distributed under Apache License 2.0. The orientation check uses
Tesseract's separate `osd.traineddata`, which the Windows build copies from
the Tesseract installation. It only reads the orientation result; recognized
text is not retained or written.
