FROM python:3.6

RUN pip install click keras pillow tensorflow

COPY no_scaling_fix_overnight /no_scaling_fix_overnight
COPY network.py /network.py
COPY docker.py /docker.py

ENTRYPOINT ["python", "/docker.py"]
