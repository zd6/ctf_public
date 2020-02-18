"""
CtF environment script and map generator

It is written to generate video rendering and script for analysis/debugging.

Original Author :
    Jacob (jacob-heglund)
Modifier/Editor :
    Seung Hyun (skim0119)
    Shamith (shamith2)
"""

import numpy as np
import csv

import os
import time
import gym
import gym_cap
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# the modules that you can use to generate the policy.
import gym_cap.heuristic as policy

import moviepy.editor as mp
from moviepy.video.fx.all import speedx

# Run Settings
total_run = 1000
max_episode_length = 150

# Environment Preparation
env = gym.make("cap-v0").unwrapped 
observation = env.reset(map_size=20,
        policy_blue=policy.roomba,
        policy_red=policy.random
    )
def rollout():
    observation = env.reset()
    episode_length = 0
    done = False
    while not done and episode_length < max_episode_length:
        yield False
        observation, reward, done, _ = env.step()
        episode_length += 1
    yield True

# Export Settings
data_dir = 'render'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
raw_dir = 'raw_videos'
video_dir = os.path.join(data_dir, raw_dir)
if not os.path.exists(video_dir):
    os.mkdir(video_dir)
min_length = 40
max_length = 120
num_success = 10  # number of rendering for success case
num_failure = 10  # number of rendering for failure case

vid_success = []
vid_failure = []

def play_episode(episode = 0):
    """
    play single episode
    """

    agent_waypoints = []
    played_map = env.get_full_state
    
    video_fn = 'episode_' + str(episode) + '.mp4'
    video_path = os.path.join(video_dir, video_fn)
    video_recorder = VideoRecorder(env, video_path)

    length = 0
    for done in rollout():
        length += 1
        video_recorder.capture_frame()

        # Optain waypoints
        waypoints = []
        for entity in env.get_team_blue.tolist() + env.get_team_red.tolist():
            waypoints.extend(entity.get_loc())
        agent_waypoints.append(waypoints)

        if done:
            break

    # Closer
    video_recorder.close()
    vid = mp.VideoFileClip(video_path)

    # Check if episode has right length played
    if length <= min_length or length >= max_length:
        return

    # Post Processing
    if env.blue_win is True and len(vid_success) < num_success:
        vid_success.append((vid, agent_waypoints, played_map))
    elif env.blue_win is False and len(vid_failure) < num_failure:
        vid_failure.append((vid, agent_waypoints, played_map))

def render_clip(frame, filename):
    """
    Render single clip with delayed frame
    """
    vid = speedx(frame, 0.1)
    video_path = os.path.join(data_dir, filename)          
    vid.write_videofile(video_path, verbose=False)
        

if __name__ == "__main__":
    # Run until enough success episodes arefound
    episode = 0
    while len(vid_success) < num_success and episode <= total_run:
        play_episode(episode)   
        episode += 1

    print(f"Requested {total_run} episode run - played {episode} run")
    print(f"{len(vid_success)} success mode saved, and {len(vid_failure)} failure mode saved")
        
    env.close()

    # Export
    for idx, (video, script, init_map) in enumerate(vid_success):
        # Export Video clip
        render_clip(video, f'run_{idx}_video.mp4')

        # Export Waypoint script
        with open(f'{data_dir}/run_{idx}_script.csv', mode='wt') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # Write field
            field = []
            for i, entity in enumerate(env.get_team_blue):
                if entity.air:
                    field.append('b_air_{}_x'.format(i))
                    field.append('b_air_{}_y'.format(i))
                else:
                    field.append('b_ground_{}_x'.format(i))
                    field.append('b_ground_{}_y'.format(i))
            for i, entity in enumerate(env.get_team_red):
                if entity.air:
                    field.append('r_air_{}_x'.format(i))
                    field.append('r_air_{}_y'.format(i))
                else:
                    field.append('r_ground_{}_x'.format(i))
                    field.append('r_ground_{}_y'.format(i))
            
            writer.writerow(field)
            for waypoint in script:
                writer.writerow(waypoint)

        # Export played map
        np.savetxt(f'{data_dir}/run_{idx}_map.txt', init_map.astype(int), delimiter=',', fmt='%i')
