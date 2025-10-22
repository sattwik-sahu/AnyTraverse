#!/usr/bin/zsh

docker run -it --privileged \
  --gpus all --runtime nvidia \
  -v /dev/bus/usb:/dev/bus/usb \
  --device-cgroup-rule='c 189:* rmw' \
  --env "DISPLAY" --env "TERM=xterm-256color" \
  --volume="${XAUTHORITY}:/root/.Xauthority:rw" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --net=host --ipc=host \
  --name=jazzy-anytraverse \
  -v /home/moonlab/sattwik/projects/vision-language-mapping/anytraverse:/home/moonlab/anytraverse:rw \
  awesome-jazzy:v1.0.1

# --env="DISPLAY" \      
#   -v $(pwd):/home/moonlab/workspace:rw \
#   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#   --volume="${XAUTHORITY}:/root/.Xauthority:rw" \              
#   -e  \
#   --device /dev/dri:/dev/dri:rw \
  
