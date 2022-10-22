# e-PEDRA
## About
PEDRA augmented with event-based vision for hybrid event + frame processing applications for drone platforms. 

The simulators for robotic interactions with the environment operate in a frame-based paradigm. Thus, only frame-based vision processing is available. Recently popular event sensors provide an asynchronous event streams instead of discrete frames. These are highly suited for high-speed and high dynamic range vision applications. However, emulating the event-based vision on closed-loop simulators requires integration of frame-to-event converters with conventional platforms. This work builds upon PEDRA (drone simulator in virtual environments) and augments it with v2e (frame-to-event conversion tool). 

## Installation
The software setup requires conflicting environments for some modules within these environments. Thus, please follow the procedure outlined below to build it using Anaconda. The hardware requirements remains the same as PEDRA and v2e. Run the following commands to build mutually compatible python libraries.
Local PC configuration: Intel i9 Processor and NVIDIA Quadro RTX 4000 GPU, CUDA 11.2
conda create -n myenv python=3.6
conda install pytorch=1.7.0 cudatoolkit=11.0 -c pytorch
conda install -c pytorch torchvision=0.8.2
pip install -r requirements_v2e_e-PEDRA.txt

## Testing
Clone PEDRA: https://github.com/aqeelanwar/PEDRA
Clone v2e: https://github.com/SensorsINI/v2e
Copy the contents of v2e to PEDRA directory. Replace files if needed.
Copy the contents of this link to unreal_envs/: https://gtvault-my.sharepoint.com/:f:/g/personal/alele9_gatech_edu/EoSFgvuxDi1BkmvQMRk0KW8Bn5HlxxtQryiY_8h76Y7Rug?e=wOCRcs

This file only contants the scripts that need to be modified in order to run v2e within PEDRA.
Therefore, copy/replace the contents of the directories in this repo to the directories with the same name in the merged directory.

## Verification:
Run - python v2e_pedra.py
Check the contants of the videos - frame_optical.avi, frame_event.avi.
The videos will contant the frame and event data for processing if the installation works correctly.
Tensorflow 1.14 and pytorch 1.7 are available for developing applications using these images.

## Acknowledgement
This tool is created as a part of C-BRIC (https://engineering.purdue.edu/C-BRIC).

## Citation:
PEDRA: Follow instructions at https://github.com/aqeelanwar/PEDRA
V2E: Follow instructions at https://github.com/SensorsINI/v2e

Please contact Ashwin Lele (alele9@gatech.edu/ ashwinlele.2009@gmail.com) in case of any queries. 
