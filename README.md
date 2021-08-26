## Python installation (non-ROS)
Install motion_primitives_py package:
 - `pip3 install -e .`

Required Python packages not installed by setup.py:
- If you don't have ROS installed and sourced: `pip3 install --extra-index-url https://rospypi.github.io/simple/ rosbag`

System packages (for animation video mp4s to be generated):
- `sudo apt-get install ffmpeg`

## C++ Installation (ROS)
Must have ROS and catkin tools already installed and sourced. Independent from/does not also install standalone python package. Graph generation is only included in python, graph search exists in both (but focus is on C++ version).

```
sudo apt-get install -y libeigen3-dev libtbb-dev libgtest-dev python3-vcstool
mkdir -p dispersion_ws/src
cd dispersion_ws
catkin init
git clone git@github.com:ljarin/motion_primitives.git
git clone git@github.com:ljarin/planning_ros_msgs.git
catkin b
```
