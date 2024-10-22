from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dpg import DPGAgent  # Import the DPGAgent
import torch
import numpy as np
import random
from datetime import datetime
import time
from plots import plot_average_queue
from torch.distributions import Categorical

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 8001, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/train.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 801, '')
flags.DEFINE_string('network', 'policy_gradient', '')  # Update network type
flags.DEFINE_string('mode', 'train', '')  # train or eval
flags.DEFINE_string('network_file', '', '')  # Weights file for policy
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_bool('use_sgd', True, 'Training with the optimizer SGD or RMSprop')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device == "cuda":
    torch.cuda.manual_seed_all(random_seed)

current_date = str(datetime.now()).split('.')[0].split(' ')[0].replace('-', '')

def main(argv):
    del argv

    env = SumoEnv(
        net_file=FLAGS.net_file,
        route_file=FLAGS.route_file,
        skip_range=FLAGS.skip_range,
        simulation_time=FLAGS.simulation_time,
        yellow_time=FLAGS.yellow_time,
        delta_rs_update_time=FLAGS.delta_rs_update_time,
        reward_fn=FLAGS.reward_fn,
        mode=FLAGS.mode,
        use_gui=FLAGS.use_gui,
    )

    start_time = time.time()

    # Initialize your DPG agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DPGAgent(input_dim, output_dim, FLAGS.gamma)

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        trajectory = []  # Collecting trajectory for policy update

        while not done:
            state = env.compute_state()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = agent.policy_network(state_tensor)
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if info['do_action'] is not None and reward is not None:
                do_action = torch.tensor(info['do_action'], device=device)
                actual_log_prob = Categorical(action_probs).log_prob(do_action)
                trajectory.append((state, info['do_action'], actual_log_prob, reward, next_state))

                # Free up the GPU memory after processing the state and actions
                del do_action

                # Free up the GPU memory for these variables regardless of action validity
            del state_tensor, action_probs
            torch.cuda.empty_cache()

            # Perform policy update after each episode
        if FLAGS.mode == 'train':
            agent.update_policy(trajectory)

        env.close()

        if FLAGS.mode == 'train':
            if episode != 0 and episode % 25 == 0:
                torch.save(agent.policy_network.state_dict(), f'weights/weights_{current_date}_{episode}.pth')

        print('i_episode:', episode)
        print(env.avg_queue[-1])

        current_time = time.time()
        print(current_time - start_time)

        # Plot and save average queue data
        if FLAGS.mode == 'train' and episode != 0 and episode % 25 == 0:
            plot_average_queue(env.avg_queue, episode, current_date)
            print(env.avg_queue)

        if FLAGS.mode == 'train' and episode != 0:
            with open("record/avg_queue.txt", "w") as file:
                file.write(','.join(map(str, env.avg_queue)))


if __name__ == '__main__':
    app.run(main)