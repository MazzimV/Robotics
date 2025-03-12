#!/bin/bash -e
###
 # @Date: 2025-02-13
 # @LastEditors: JJQ jj1623@ic.ac.uk
 # @LastEditTime: 2025-02-13
 # @FilePath: /ELEC70120/workspace/entrypoint.sh
 # @Description: 
### 

USER_ID=$(id -u)
GROUP_ID=$(id -g)

sudo usermod -u $USER_ID -o -m -d /home/developer developer > /dev/null 2>&1
sudo groupmod -g $GROUP_ID developer > /dev/null 2>&1
# sudo chown -R developer:developer /workspace
echo "Changing ownership of /workspace except .git folder"
sudo find /workspace \
    -path /workspace/.git -prune -o \
    -exec chown developer:developer {} +

ln -sfn /home/developer/.vscode /workspace/.vscode

rm -f /workspace/compile_flags.txt || true
sed -e 's@\$ROS_DISTRO@'"$ROS_DISTRO"'@' /home/developer/compile_flags.txt > /workspace/compile_flags.txt

ln -sfn /workspace /home/developer/workspace

source /opt/ros/$ROS_DISTRO/setup.bash

mkdir -p /workspace/src && cd /workspace/src && catkin_init_workspace || true

cd /workspace && catkin build && echo "source /workspace/devel/setup.bash" >> ~/.bashrc

echo "Setting up ROS environment variables..."

cd /home/developer

exec $@