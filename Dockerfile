FROM python:3.6

RUN pip install click keras pillow tensorflow

COPY docker.py /docker.py
COPY network.py /network.py
COPY no_scaling_fix_overnight /no_scaling_fix_overnight

ENTRYPOINT ["python", "/docker.py"]
