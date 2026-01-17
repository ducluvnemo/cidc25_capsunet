FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git wget && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/algorithm
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /opt/algorithm
RUN groupadd -r algouser && useradd -r -g algouser algouser && chown -R algouser:algouser /opt/algorithm
USER algouser
ENTRYPOINT ["python3", "-m", "challenge.predict"]
