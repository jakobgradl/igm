# docker build --no-cache -t igm --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
#
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update

# Add user
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID igmuser
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID igmuser

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Core linux dependencies. 
RUN apt-get install -y nano git

CMD ["bash"]

RUN pip3 install -U pip

# RUN pip3 install numpy

USER igmuser 

WORKDIR /home/igmuser/
  
RUN git clone https://github.com/jouvetg/igm.git

WORKDIR /home/igmuser/igm/

RUN pip install -e .

WORKDIR /home/igmuser/