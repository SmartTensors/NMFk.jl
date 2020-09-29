LABEL maintainer="monty velimir.vesselinov@gmail.com"

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p /opt/src
WORKDIR /opt/src

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        wget \
        unzip \
        git \
  && apt-add-repository ppa:ubuntugis/ubuntugis-unstable \
  && apt-add-repository ppa:deadsnakes/ppa \
  && apt-get update -y \
  && apt install -y --no-install-recommends \
        build-essential \
        python3.8 \
        python3.8-dev \
        python3-pip \
        python3-setuptools \
        gdal-bin=3.0.4+dfsg-1~bionic0 \
        libgdal-dev \
    && rm -rf /usr/bin/python /usr/bin/python3 \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && ln -s /usr/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && pip3 install -U pip==20.2.3 setuptools==50.3.0 \
    && rm -rf ~/.cache/pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/src/requirements.txt
RUN pip install --ignore-installed -r requirements.txt \
    && rm -rf ~/.cache/pip

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.2-linux-x86_64.tar.gz
RUN tar xvzf julia-1.5.2-linux-x86_64.tar.gz
RUN rm -f julia-1.5.2-linux-x86_64.tar.gz
RUN ln -sf /opt/src/julia-1.5.2/bin/julia /usr/bin
