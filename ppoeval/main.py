from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.ppo import PPOAgent
from datetime import datetime
import torch
import numpy as np
import os
import time
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 4001, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/eval6.rou.xml', '')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 1, 'Number of evaluation episodes')
flags.DEFINE_string('network_file', 'weights/ppo.pth', 'Path to the trained model weights')

device = "cuda" if torch.cuda.is_available() else "cpu"

random_seed = 40
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device == "cuda":
    torch.cuda.manual_seed_all(random_seed)

def main(argv):
    del argv

    # Initialize the environment
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  reward_fn=FLAGS.reward_fn,
                  mode='eval',  # Ensure evaluation mode
                  use_gui=FLAGS.use_gui)

    # Initialize PPO Agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PPOAgent(state_dim=input_dim,
                     action_dim=output_dim,
                     lr=0.0003,  # Learning rate is arbitrary during evaluation as no policy update occurs
                     gamma=0.95,
                     clip_epsilon=0.2,
                     gae_lambda=0.95,
                     batch_size=64,
                     update_epochs=10,
                     network_file=FLAGS.network_file)

    # Load model weights
    if FLAGS.network_file and os.path.exists(FLAGS.network_file):
        agent.policy_net.load_state_dict(torch.load(FLAGS.network_file, map_location=device))
        print(f"Loaded weights from {FLAGS.network_file}")
    else:
        raise ValueError(f"Invalid weights file path: {FLAGS.network_file}")

    # Start evaluation
    total_avg_queue = []  # To record the average queue length for each episode
    start_time = time.time()

    for episode in range(FLAGS.num_episodes):
        # Reset the environment
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        trajectory = []  # Collecting trajectory for policy update

        while not done:
            state = env.compute_state()
            # Use PPOAgent to select an action
            action, _ = agent.select_action(state)

            # Execute the action
            next_state, reward, done, info = env.step(action)
            if info['do_action'] is None or reward is None:
                invalid_action = True
                continue
        # Record the average queue length for the current episode
        avg_queue = env.avg_queue[-1] / 8
        total_avg_queue.append(avg_queue)

        print(f"Episode {episode + 1}: Average Queue = {avg_queue}")

        # Close the environment
        env.close()

    # Calculate and print the overall average queue length
    overall_avg_queue = np.mean(total_avg_queue)
    print(f"Overall Average Queue Length over {FLAGS.num_episodes} episodes: {overall_avg_queue}")

    # Save the evaluation results to a file
    if not os.path.exists("record"):
        os.makedirs("record")
    with open("record/eval_avg_queue.txt", "w") as file:
        file.write(f"Average Queue Lengths: {total_avg_queue}\n")
        file.write(f"Overall Average Queue Length: {overall_avg_queue}\n")

    # Print total time taken
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    app.run(main)