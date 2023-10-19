FROM ic-registry.epfl.ch/mlo/pytorch:latest
MAINTAINER Marco Bondaschi <marco.bondaschi@epfl.ch>


# install some necessary tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential
RUN apt-get install -y ca-certificates 
#RUN apt-get install -y software-properties-common
RUN apt-get install -y pkg-config
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN pip3 install -U torch torchvision torchaudio
RUN pip install tiktoken
RUN apt-get install -y \
        inkscape \
        texlive-latex-extra \
        dvipng \
        texlive-full \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libcurl3-dev \
        libfreetype6-dev \
        libzmq3-dev \
        libcupti-dev \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        locales
RUN apt-get install -y \
        rsync \
        cmake \
        g++ \
        swig \
        vim \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        tmux
RUN git clone https://github.com/GAMPTeam/vampyre.git
RUN pip install -e /vampyre
RUN pip install wandb --upgrade
#RUN apt-get install -y openssh-server
# install good vim.
#RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
#RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
# ENV SHELL=/bin/bash \
#     NB_USER=bondasch \
#     NB_UID=223002 \
#     NB_GROUP=MLO-unit \
#     NB_GID=11218
# ENV HOME=/home/$NB_USER

# RUN groupadd $NB_GROUP -g $NB_GID
# RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
#     echo "${NB_USER}:${NB_USER}" | chpasswd && \
#     usermod -aG sudo,adm,root ${NB_USER}
# RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# The user gets passwordless sudo
#RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

###### switch to user and compile test example.
#USER ${NB_USER}

###### switch to root
# expose port for ssh and start ssh service.
#EXPOSE 22
# expose port for notebook.
#EXPOSE 8888
# expose port for tensorboard.
#EXPOSE 6666
