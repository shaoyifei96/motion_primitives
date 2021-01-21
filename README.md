Install motion_primitives_py package:
 - `pip3 install -e .`

Required Python packages not installed by setup.py:
- https://github.com/jpaulos/opt_control (be sure to follow the installation instructions in the python subdirectory)
- If you don't have ROS installed and sourced: `pip3 install --extra-index-url https://rospypi.github.io/simple/ rosbag`

System packages (for animation video mp4s to be generated):
- `sudo apt-get install ffmpeg`