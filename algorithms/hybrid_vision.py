# Author: Aqeel Anwar(ICSRL)
# Created: 2/19/2020, 8:39 AM
# Email: aqeel.anwar@gatech.edu

import sys, cv2
#import nvidia_smi
from network.agent import PedraAgent
from unreal_envs.initial_positions import *
from os import getpid
from network.Memory import Memory
from aux_functions import *
import os
from util.transformations import euler_from_quaternion
from configs.read_cfg import read_cfg, update_algorithm_cfg
import csv
import matplotlib.pylab as plt
import matplotlib.patches as patches
from v2e_function import *


########## slomo libraries ##########
from v2e.slomo import SuperSloMo
import glob
import argparse
import importlib
from pathlib import Path
import os

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory, TemporaryFile
from engineering_notation import EngNumber  as eng # only from pip
from tqdm import tqdm


import v2e.desktop as desktop
from v2e.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2e.v2e_args import v2e_args, write_args_info, v2e_check_dvs_exposure_args
from v2e.v2e_args import NO_SLOWDOWN
from v2e.renderer import EventRenderer, ExposureMode
from v2e.slomo import SuperSloMo
from v2e.emulator import EventEmulator
from v2e.v2e_utils import inputVideoFileDialog
import logging



def frame_event_gen(frame1,frame2):
    frame1 = np.where(frame1<5,frame1/5*math.log(5),frame1)
    frame1 = np.where(frame1>=5,math.log(frame1),frame1)

    frame2 = np.where(frame2<5,frame2/5*math.log(5),frame2)
    frame2 = np.where(frame2>=5,math.log(frame2),frame2)
    return event_frame

def hybrid_vision(cfg, env_process, env_folder):

    algorithm_cfg = read_cfg(config_filename='configs/hybrid_vision.cfg', verbose=True)
    algorithm_cfg.algorithm = cfg.algorithm

    
    client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode, num_agents=cfg.num_agents)
    initial_pos = old_posit.copy()
    reset_array, reset_array_raw, level_name, crash_threshold = initial_positions(cfg.env_name, initZ, cfg.num_agents)
    #print(reset_array_raw,'yyyy',cfg.num_agents)
    process = psutil.Process(getpid())
    
    # Load PyGame Screen
    screen = pygame_connect(phase=cfg.mode)

    fig_z = []
    fig_nav = []
    debug = False
    # Generate path where the weights will be saved
    cfg, algorithm_cfg = save_network_path(cfg=cfg, algorithm_cfg=algorithm_cfg)
    current_state = {}
    new_state = {}
    posit = {}
    name_agent_list = []
    agent = {}
    # Replay Memory for RL
    
    for drone in range(cfg.num_agents):
        name_agent = "drone" + str(drone)
        
        name_agent_list.append(name_agent)
        agent[name_agent] = PedraAgent(algorithm_cfg, client, name = name_agent + 'DQN', vehicle_name = name_agent)
        
    
        env_cfg = read_cfg(config_filename=env_folder+'config.cfg')
        nav_x = []
        nav_y = []
        altitude = {}
        altitude[name_agent] = []
        p_z,f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_infer(env_cfg=env_cfg, client=client, env_folder=env_folder)
        nav_text = ax_nav.text(0, 0, '')

    reset_to_initial(0, reset_array_raw, client, vehicle_name="drone0")
    old_posit["drone0"] = client.simGetVehiclePose(vehicle_name="drone0")
    print('current position', old_posit["drone0"])
    
    # Select initial position
    # Initialize variables
    
    iter = 1
    step = 0
    action_array = 4*np.ones((10))
    action_array = action_array.astype(int)

    # num_collisions = 0
    episode = {}
    active = True

    print_interval = 1
    automate = True
    choose = False
    print_qval = False
    last_crash = {}
    ret = {}
    distance = {}
    num_collisions = {}
    level = {}
    level_state = {}
    level_posit = {}
    times_switch = {}
    last_crash_array ={}
    ret_array ={}
    distance_array ={}
    epi_env_array = {}
    log_files = {}

    # If the phase is inference force the num_agents to 1
    hyphens = '-' * int((80 - len('Log files')) / 2)
    print(hyphens + ' ' + 'Log files' + ' ' + hyphens)
    ignore_collision = False
    
    for name_agent in name_agent_list:
        #print(name_agent, name_agent_list)
        ret[name_agent] = 0
        num_collisions[name_agent] = 0
        last_crash[name_agent] = 0
        level[name_agent] = 0
        episode[name_agent] = 0
        level_state[name_agent] = [None] * len(reset_array[name_agent])
        level_posit[name_agent] = [None] * len(reset_array[name_agent])
        times_switch[name_agent] = 0
        last_crash_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        ret_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        distance_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]))
        epi_env_array[name_agent] = np.zeros(shape=len(reset_array[name_agent]), dtype=np.int32)
        distance[name_agent] = 0
        # Log file
        log_path = 'algorithms/log.txt'
        print('oooooo',log_path)
        #print("Log path: ", log_path)
        log_files = open(log_path, 'w')
    
    
    print_orderly('Simulation begins', 80)
    

    # save_posit = old_posit
    iter = -1
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_event = cv2.VideoWriter('event_frame.avi',fourcc,10,(640,480))
    video_frame = cv2.VideoWriter('optical_frame.avi',fourcc,10,(640,480))
    video_dvs = cv2.VideoWriter('optical_dvs.avi',fourcc,10,(640,480))
    
    def get_args():
        parser = argparse.ArgumentParser(
            description='v2e: generate simulated DVS events from video.',
            epilog='Run with no --input to open file dialog', allow_abbrev=True,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = v2e_args(parser)
        parser.add_argument(
            "--rotate180", type=bool, default=False,
            help="rotate all output 180 deg.")
        # https://kislyuk.github.io/argcomplete/#global-completion
        # Shellcode (only necessary if global completion is not activated -
        # see Global completion below), to be put in e.g. .bashrc:
        # eval "$(register-python-argcomplete v2e.py)"
        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        return args
    
    args=get_args()
    slowdown_factor=NO_SLOWDOWN
    output_folder: str = os.getcwd()  + "\\output\\tennis\\"
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview = False
    batch_size = 4
    
    slomo = SuperSloMo(
                model=os.getcwd()+"\\input\\SuperSloMo39.ckpt", auto_upsample=False, upsampling_factor=10,
                video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo,
                preview=preview, batch_size=batch_size)
    
    while active:
        try:
            active, automate, algorithm_cfg, client = check_user_input(active, automate, agent[name_agent], client, old_posit[name_agent], initZ, fig_z, fig_nav, env_folder, cfg, algorithm_cfg)
            #print()
            if automate:

                
                if cfg.mode == 'infer':
                    for drone in range(cfg.num_agents):
                        
                        name_agent = "drone" + str(drone)
                        
                        agent_state = agent[name_agent].GetAgentState()
                        posit[name_agent] = client.simGetVehiclePose(vehicle_name=name_agent)
                        distance[name_agent] = distance[name_agent] + np.linalg.norm(np.array([old_posit[name_agent].position.x_val-posit[name_agent].position.x_val,old_posit[name_agent].position.y_val-posit[name_agent].position.y_val]))
                        
                        quat = (posit[name_agent].orientation.w_val, posit[name_agent].orientation.x_val, posit[name_agent].orientation.y_val, posit[name_agent].orientation.z_val)
                        yaw = euler_from_quaternion(quat)[2]

                        x_val = posit[name_agent].position.x_val
                        y_val = posit[name_agent].position.y_val
                        z_val = posit[name_agent].position.z_val
                        
                        if name_agent == "drone0":
                            
                            iter = iter + 1
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))
                            video_frame.write(camera_image.astype(np.uint8))
                            
                            responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)], vehicle_name= name_agent)
                            response = responses[0]
                            depth = np.array(response.image_data_float, dtype=np.float32)
                            depth = depth.reshape(response.height, response.width)
                            depth = np.array(depth * 255, dtype=np.uint8)
                            depth = depth.astype(float)/255
                            depth = cv2.resize(depth,(640,480))

                            if iter > 4:
                                dvs_image = v2e_function(slomo)
                                dvs_image = 255*np.repeat(dvs_image[:, :, np.newaxis], 3, axis=2)
                                print(np.shape(dvs_image), dvs_image)
                                video_event.write(dvs_image.astype(np.uint8))
                                cv2.imshow('a', dvs_image)
                                cv2.waitKey(30)

                            video_dvs = cv2.VideoWriter('optical_dvs.avi',fourcc,10,(640,480))
                            old_posit[name_agent] = posit[name_agent]

                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))
                            video_frame.write(camera_image.astype(np.uint8))
                            
                            current_state[name_agent] = agent[name_agent].get_state1()
                            action, action_type, algorithm_cfg.epsilon, qvals, step = policy(1, current_state[name_agent], iter,
                                                                              algorithm_cfg.epsilon_saturation, 'inference',
                                                                              algorithm_cfg.wait_before_train, algorithm_cfg.num_actions, agent[name_agent],action_array, step)
                            action_word = translate_action(action, algorithm_cfg.num_actions)
                            
                            prey_speed = agent[name_agent].take_action(action, iter, algorithm_cfg.num_actions, SimMode=cfg.SimMode)
                            old_posit[name_agent] = posit[name_agent]
                            
                            camera_image = get_MonocularImageRGB(client, vehicle_name=name_agent)
                            video_dvs.write(camera_image.astype(np.uint8))
                            video_frame.write(camera_image.astype(np.uint8))
                            

                            
                    s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f} Predicted Action: {:<8s}  '.format(
                        x_val, y_val, z_val, yaw, action_word
                    )

                    log_files.write(s_log + '\n')



        except Exception as e:
            if str(e) == 'cannot reshape array of size 1 into shape (0,0,3)':
                print('Recovering from AirSim error')
                client, old_posit, initZ = connect_drone(ip_address=cfg.ip_address, phase=cfg.mode,
                                                         num_agents=cfg.num_agents)

                agent[name_agent].client = client
                video_dvs.release()
            else:
                print('------------- Error -------------')
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(exc_obj)
                automate = False
                print('Hit r and then backspace to start from this point')


