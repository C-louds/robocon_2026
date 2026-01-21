launching the gazebo:
`ros2 launch bcr_bot ign.launch.py`

relay the bcr_bot and odom signal
`ros2 run topic_tools relay /bcr_bot/scan /scan`
in other terminal
`ros2 run topic_tools relay /bcr_bot/odom /odom`

rviz launch
`ros2 launch bcr_bot rviz.launch.py`

slam and stuff
`ros2 launch slam_toolbox online_async_launch.py`
