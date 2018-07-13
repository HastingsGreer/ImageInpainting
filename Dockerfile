FROM python:3.6

RUN pip install click keras pillow tensorflow

COPY docker.py /docker.py
COPY network.py /network.py
COPY nvidia_monday /nvidia_monday

ENTRYPOINT ["python", "/docker.py"]
