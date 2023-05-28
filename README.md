# Pose_Estimation_Applications
Here are some applications based of on [Pose_Estimation](https://github.com/REZ3LIET/Pose_Estimations) repository which deal wiith tracking body poses, hand poses and detecting faces. Some basic applications done are [Volume_Control](./Volume_Control/), [Finger_Counter](./Finger_Counter/), [Fitness_Tracker](./Fitness_Tracker/).

For more details, each project has its own README file.

## Installation and Setup
- It is recommended to have a virtual environment setup. This can be done using the following commands
    - Create an empty directory named PE_Apps
    - Create a virtual environmaent
    ```bash
    mkdir PE_Apps
    cd PE_Apps
    python3 -m venv pe_apps
    ```
    - Activate the virtual environment
    ```bash
    # To activate in Linux
    source pe_apps/bin/activate

    # To activate in Windows
    ./pe_apps/Scripts/activate
    ```

- To use this repository you can use the basic `git clone` command inside the directory  
```bash
git clone https://github.com/REZ3LIET/Pose_Estimation_Applications.git
```

- Use `pip` to install the required packages
```bash
pip3 install -r requirements.txt
```
## List of Applications

Here is the list list of projects which utilise the [Pose_Estimation](https://github.com/REZ3LIET/Pose_Estimations) repository.
- [Finger_Counter](./Finger_Counter/)
- [Fitness_Tracker](./Fitness_Tracker/)
    - [Push_Up_Counter](./Fitness_Tracker/Push_Up_Counter/)
- [Volume_Control](./Volume_Control/)
