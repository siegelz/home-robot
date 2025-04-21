FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2024-ubuntu22.04-v2

# install baseline agent requirements
RUN /bin/bash -c "\
    . activate home-robot \
    && cd home-robot \
    && git submodule update --init --recursive src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet \
    && pip install -e src/third_party/detectron2 \
    && pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
    && pip install -e src/home_robot \
    "

# download pretrained Detic checkpoint
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate

# download pretrained skills
RUN /bin/bash -c "\
    mkdir -p home-robot/data/checkpoints \
    && cd home-robot/data/checkpoints \
    && wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023_v0.2.zip \
        -O ovmm_baseline_home_robot_challenge_2023_v0.2.zip \
        --no-check-certificate \
    && unzip ovmm_baseline_home_robot_challenge_2023_v0.2.zip -d ovmm_baseline_home_robot_challenge_2023_v0.2 \
    && rm ovmm_baseline_home_robot_challenge_2023_v0.2.zip \
    "

# add baseline agent code
ADD eval_baselines_agent.py /home-robot/projects/habitat_ovmm/agent.py

# add submission script
ADD scripts/submission.sh /home-robot/submission.sh

# set evaluation type to remote
ENV AGENT_EVALUATION_TYPE remote

# additional command line arguments for local evaluations (empty for remote evaluation)
ENV LOCAL_ARGS ""


# =========== OGN SETUP  =========== #

# cmake already installed.

# git checkout tags/v0.1.5; \
# Setup habitat-sim
RUN /bin/bash -c "git clone https://github.com/roma0615/habitat-sim.git; \
    cd habitat-sim; \
    . activate home-robot; \
    git checkout cos435_main; \
    pip install -r requirements.txt; \
    python setup.py install --headless --with-cuda --parallel 4"

# Install challenge specific habitat-api
RUN /bin/bash -c "git clone https://github.com/facebookresearch/habitat-api.git; \
    . activate home-robot; \
    cd habitat-api; \
    git checkout tags/v0.2.4; \
    pip install -e habitat-lab"
RUN /bin/bash -c "cd habitat-api; \
    wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip; \
    unzip habitat-test-scenes.zip"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# Install project specific packages
RUN /bin/bash -c "apt-get update; apt-get install -y libsm6 libxext6 libxrender-dev;"

# pip install --upgrade cython numpy; \
RUN /bin/bash -c ". activate home-robot; \
    conda install scikit-fmm scikit-image scikit-image scikit-learn -y; \
    pip install opencv-python; \
    pip install matplotlib seaborn imageio ifcfg"

# Install pytorch and torch_scatter
RUN /bin/bash -c ". activate home-robot; conda install cudatoolkit=11.7 -c pytorch -y; "

# Add datasets to docker container

# =========== END OGN SETUP  =========== #

# clone the thing
RUN /bin/bash -c "git clone https://github.com/roma0615/Object-Goal-Navigation.git; "

# CMD /bin/bash -c "\
#     . activate home-robot \
#     && cd /Object-Goal-Navigation \
#     && python main.py"

# run submission script
# TODO change
CMD /bin/bash -c "\
    . activate home-robot \
    && cd /home-robot \
    && export PYTHONPATH=/evalai_remote_evaluation:$PYTHONPATH \
    && bash submission.sh $LOCAL_ARGS \
    "
