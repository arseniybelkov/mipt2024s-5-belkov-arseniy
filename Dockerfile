FROM python:3.11

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./src /usr/src/app/
ADD ./requirements.txt /usr/src/app/requirements.txt


RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install opencv-python-headless

CMD ["bash"]