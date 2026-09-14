FROM python:3.11-slim

WORKDIR /app

# Some dependencies (tensorflow, via deepface) are large (500MB+) wheels;
# default pip timeouts are too aggressive for slower/unstable networks and
# cause misleading failures partway through a big download.
ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

# deepface pulls in plain opencv-python as a transitive dependency, which
# conflicts with opencv-python-headless (both install to the same cv2/
# namespace, corrupting native submodules like cv2.CascadeClassifier).
# This is deliberately split into separate RUN steps — chaining them with
# `&&`/`||` onto the main install above previously caused a failure in
# THAT step to be silently swallowed by this step's own error tolerance,
# masking real install failures. Each step here fails loudly on its own.
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true
# No --no-deps here: this needs to reinstall numpy and any other real
# dependency opencv-python-headless requires, not skip them.
RUN pip install --no-cache-dir --force-reinstall --retries 10 "opencv-python-headless>=4.10.0,<5.0.0"
RUN python3 -c "import cv2; assert hasattr(cv2, 'CascadeClassifier'), 'opencv install broken'; print('opencv OK:', cv2.__version__)"

COPY src/ ./src/

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
