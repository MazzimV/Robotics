#!/bin/bash
###
 # @Date: 2025-02-13
 # @LastEditors: JJQ jj1623@ic.ac.uk
 # @LastEditTime: 2025-02-13
 # @FilePath: /ELEC70120/workspace/build_run.sh
 # @Description: 
### 

# 设定镜像名称
IMAGE_NAME="devrt/workspace_real"

# 1. 构建 Docker 镜像
echo "Building Docker image: $IMAGE_NAME..."
docker build -f Dockerfile -t $IMAGE_NAME .

# 2. 启动 docker-compose
echo "Starting docker-compose..."
docker-compose up -d

echo "Done!"
