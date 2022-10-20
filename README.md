# e-PEDRA
PEDRA augmented with event-based vision

This project is intended towards hybrid (event + frame) vision applications for drone platforms. This was built during the hybrid SNN + CNN target tracking project: .

The simulators for robotic interactions with the environment operate in a frame-based paradigm. Thus, only frame-based vision sensing and processing is applicable. Recently popular event sensors provide an asynchronous event streams instead of discrete frames. These are highly suited for high-speed and high dynamic range visual sensing. However, emulating the event-based vision of closed-loop simulators requires integration of frame-to-event converters.

This work builds upon PEDRA (drone simulator in virtual environments) and augments it with v2e (frame-to-event conversion tool). 

Installation:
The hardware requirements are similar to PEDRA. The software versions need to followed accurately to avoid version mismatch in python.

Verification:
Plot event-accumulated frames from drone motion
Plot frames captures from the onboard camera

This allows the user to build tensorflow applications in the backend.

This tool is created as a part of C-BRIC ().

Citation:
PEDRA: Follow instructions at
V2E: Follow instructions at
E-PEDRA:

Please contact Ashwin Lele (ashwinlele.2009@gatech.edu) for any queries. 
