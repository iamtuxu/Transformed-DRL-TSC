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
flags.DEFINE_float('simulation_time', 4001, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/eval2.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 10, '')
flags.DEFINE_string('network', 'policy_gradient', '')  # Update network type
flags.DEFINE_string('mode', 'eval', '')  # train or eval
flags.DEFINE_string('network_file', 'weights/state2.pth', '')  # Weights file for policy
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_bool('use_sgd', True, 'Training with the optimizer SGD or RMSprop')

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_min_index(state):
    if not state:
        return None  # 如果列表为空，返回None

    min_index = 0
    min_value = state[0]

    for i in range(1, len(state)):
        if state[i] < min_value:
            min_value = state[i]
            min_index = i

    return min_index

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

    # Initialize your DPG agent with possible network pre-trained weights
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DPGAgent(input_dim, output_dim, gamma=FLAGS.gamma, network_file=FLAGS.network_file)


    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        trajectory = []  # Collecting trajectory for policy update

        while not done:
            state = env.compute_state()

            # Get probabilities for all actions
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = agent.policy_network(state_tensor)

            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            if info['do_action'] is None or reward is None:
                invalid_action = True
                continue
            invalid_action = False

            # # Get the log probability for the action that was actually taken
            # do_action = torch.tensor(info['do_action'], device=device)
            # actual_log_prob = Categorical(action_probs).log_prob(do_action)
            #
            # # Store transition with log probability only if reward is not None
            # if reward is not None:
            #     trajectory.append((state, info['do_action'], actual_log_prob, reward, next_state))

        # Perform policy update after each episode
        if FLAGS.mode == 'train':
            agent.update_policy(trajectory)

        env.close()

        if FLAGS.mode == 'train':
            if episode != 0 and episode % 100 == 0:
                torch.save(agent.policy_network.state_dict(), f'weights/weights_{current_date}_{episode}.pth')

        print('i_episode:', episode)
        print(env.avg_queue[-1])
        print(f'Episode {episode} CO2 emissions: {env.total_co2_emission} grams')


        current_time = time.time()
        print(current_time - start_time)

        # Plot and save average queue data
        if FLAGS.mode == 'train' and episode != 0 and episode % 20 == 0:
            plot_average_queue(env.avg_queue, episode, current_date)
            print(env.avg_queue)

        if FLAGS.mode == 'train' and episode != 0:
            with open("record/avg_queue.txt", "w") as file:
                file.write(','.join(map(str, env.avg_queue)))

if __name__ == '__main__':
    app.run(main)


