FROM pytorch/pytorch:1.13.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/algorithm

COPY . /opt/algorithm

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "challenge.predict"]
