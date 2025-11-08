## Install Tesseract OCR
* Download and install Tesseract OCR from: https://github.com/tesseract-ocr/tesseract/releases
* Make sure to add Tesseract OCR dir containing `tesseract.exe` to your system's PATH environment variable.
* Verify the installation by running `tesseract --version` in your command line or terminal

## Install Poppler
* Download and install Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
* Add the `Library/bin` folder inside the Poppler directory to your system's PATH environment variable.
* Verify the installation by running `pdftoppm -h` in your command line or terminal

## Install Python Dependencies
```bash
conda create -n ocr_search_env python=3.13 -y \
&& conda activate ocr_search_env \
&& pip install -r requirements.txt
```
