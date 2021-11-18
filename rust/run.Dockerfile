FROM jjanzic/docker-python3-opencv

WORKDIR /root/comp245

RUN export DEBIAN_FRONTEND=noninteractive
RUN export OpenCV_DIR="/usr/local/lib/cmake/opencv4"
RUN export LD_LIBRARY_PATH="/usr/local/lib"

RUN apt update
RUN apt install clang libclang-dev ninja-build -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get -qq autoremove
RUN apt-get -qq clean

COPY --chown=1000:1000 ./src ./src
COPY --chown=1000:1000 ./Cargo.toml ./Cargo.toml
COPY --chown=1000:1000 ./Cargo.lock ./Cargo.lock
RUN /root/.cargo/bin/cargo fetch