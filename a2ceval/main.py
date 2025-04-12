from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.ac import A2CAgent  # Import the A2CAgent instead of ACAgent
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
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/eval4.rou.xml', '')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 1, '')
flags.DEFINE_string('network', 'policy_gradient', '')  # Update network type
flags.DEFINE_string('mode', 'eval', '')  # train or eval
flags.DEFINE_string('network_file', 'weights/a2c.pth', '')  # Weights file for policy
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_bool('use_sgd', True, 'Training with the optimizer SGD or RMSprop')

device = "cuda" if torch.cuda.is_available() else "cpu"


random_seed = 40
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device == "cuda":
    torch.cuda.manual_seed_all(random_seed)

current_date = str(datetime.now()).split('.')[0].split(' ')[0].replace('-', '')

import torch
import time
from torch.distributions import Categorical

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

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = A2CAgent(input_dim, output_dim, gamma=FLAGS.gamma)

    if FLAGS.mode == 'eval' and FLAGS.network_file:
        agent.policy_network.load_state_dict(torch.load(FLAGS.network_file, map_location=device))
        print(f'{FLAGS.network_file}')

    for episode in range(FLAGS.num_episodes):
        state = env.reset()
        done = False

        while not done:
            state = env.compute_state()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = agent.policy_network(state_tensor)
            action, _ = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            if info['do_action'] is None or reward is None:
                continue


            do_action = torch.tensor(info['do_action'], device=device)
            actual_log_prob = Categorical(action_probs).log_prob(do_action)

            # Update agent with the current step's data
            if FLAGS.mode == 'train':
                agent.update(state, actual_log_prob, reward, next_state, done)

            state = next_state

        env.close()

        print(f'Episode: {episode}, Avg Queue Length: {env.avg_queue[-1] / 8}')



        if FLAGS.mode == 'train' and episode != 0 and episode % 25 == 0:

            torch.save(agent.policy_network.state_dict(), f'weights/weights_{episode}.pth')
            print(f'weights/weights_{episode}.pth')


            plot_average_queue(env.avg_queue, episode, current_date)
            print(env.avg_queue)


        if FLAGS.mode == 'train' and episode != 0:
            with open("record/avg_queue.txt", "w") as file:
                file.write(','.join(map(str, env.avg_queue)))

if __name__ == '__main__':
    app.run(main)