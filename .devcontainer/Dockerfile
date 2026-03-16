FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    git \
    git-lfs \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY evaluation/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt tensorflow soundfile

RUN git clone --recursive https://github.com/xiongyihui/python-webrtc-audio-processing.git /tmp/webrtc \
    && cd /tmp/webrtc \
    && find . -name '*_win.cc' -delete \
    && pip install --no-cache-dir . \
    && rm -rf /tmp/webrtc
