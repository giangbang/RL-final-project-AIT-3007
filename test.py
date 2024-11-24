from magent2.environments import battle_v4
import torch
from torch_model import QNetwork
import numpy as np
import os
import cv2
def test():
   env = battle_v4.env(map_size=45, render_mode="rgb_array")
   vid_dir = "video"
   os.makedirs(vid_dir, exist_ok=True)
   fps = 35
   frames = []
    # Load trained models
   blue_network = QNetwork(
       env.observation_space("blue_0").shape,
       env.action_space("blue_0").n
   )
   blue_network.load_state_dict(
       torch.load("blue.pt", map_location="cpu")
   )
    red_network = QNetwork(
       env.observation_space("red_0").shape,
       env.action_space("red_0").n
   )
   red_network.load_state_dict(
       torch.load("red.pt", map_location="cpu")
   )
    # Statistics
   total_episodes = 100
   blue_wins = 0
   red_wins = 0
   draws = 0
    for episode in range(total_episodes):
       env.reset()
       blue_agents_alive = True
       red_agents_alive = True
       
       for agent in env.agent_iter():
           observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
               action = None
               if agent.startswith("blue"):
                   blue_agents_alive = False
               elif agent.startswith("red"):
                   red_agents_alive = False
           else:
               agent_handle = agent.split("_")[0]
               observation = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0)
               
               with torch.no_grad():
                   if agent_handle == "blue":
                       q_values = blue_network(observation)
                       action = torch.argmax(q_values, dim=1).item()
                   elif agent_handle == "red":
                       q_values = red_network(observation)
                       action = torch.argmax(q_values, dim=1).item()
            env.step(action)
            # Record video for the first episode
           if episode == 0 and agent == "blue_0":
               frames.append(env.render())
            # Check if episode is done
           if not blue_agents_alive and not red_agents_alive:
               draws += 1
               break
           elif not blue_agents_alive:
               red_wins += 1
               break
           elif not red_agents_alive:
               blue_wins += 1
               break
        print(f"Episode {episode + 1} completed")
    # Save video of first episode
   if frames:
       height, width, _ = frames[0].shape
       out = cv2.VideoWriter(
           os.path.join(vid_dir, "blue_vs_red.mp4"),
           cv2.VideoWriter_fourcc(*"mp4v"),
           fps,
           (width, height),
       )
       for frame in frames:
           frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           out.write(frame_bgr)
       out.release()
    # Print results
   print("\nTest Results:")
   print(f"Total Episodes: {total_episodes}")
   print(f"Blue Wins: {blue_wins} ({blue_wins/total_episodes*100:.2f}%)")
   print(f"Red Wins: {red_wins} ({red_wins/total_episodes*100:.2f}%)")
   print(f"Draws: {draws} ({draws/total_episodes*100:.2f}%)")
    env.close()
if __name__ == "__main__":
   test()