FROM devrt/ros-devcontainer-vscode:noetic-desktop

SHELL ["/bin/bash", "-c"]
# 设置非交互模式，避免 tzdata 配置弹出
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update

# 更新 package list 并安装 image-transport-plugins
RUN apt-get update && \
    apt-get install -y ros-noetic-image-transport-plugins && \
    rm -rf /var/lib/apt/lists/*
# 
# CMD [ "sudo", "-E", "/usr/local/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

