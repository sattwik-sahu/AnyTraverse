sudo apt update
sudo apt install -y ros-humble-rosbridge-server
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
