from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.ppo import PPOAgent
from datetime import datetime
import torch
import time
from torch.distributions import Categorical
import numpy as np
from plots import plot_average_queue
import os
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 4001, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/train.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 401, '')
flags.DEFINE_string('mode', 'train', '')  
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_float('lr', 0.00025, 'learning rate')
flags.DEFINE_float('clip_epsilon', 0.2, 'PPO clipping epsilon')
flags.DEFINE_float('gae_lambda', 0.95, 'GAE lambda for advantage estimation')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('update_epochs', 10, 'Number of epochs for PPO update')
flags.DEFINE_string('network_file', '', '')

device = "cuda" if torch.cuda.is_available() else "cpu"

current_date = str(datetime.now()).split('.')[0].replace('-', '')


def main(argv):
    del argv
    # 初始化环境
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  reward_fn=FLAGS.reward_fn,
                  mode=FLAGS.mode,
                  use_gui=FLAGS.use_gui)

    # 初始化 PPO Agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = PPOAgent(state_dim=input_dim,
                     action_dim=output_dim,
                     lr=FLAGS.lr,
                     gamma=FLAGS.gamma,
                     clip_epsilon=FLAGS.clip_epsilon,
                     gae_lambda=FLAGS.gae_lambda,
                     batch_size=FLAGS.batch_size,
                     update_epochs=FLAGS.update_epochs,
                     network_file=FLAGS.network_file)


    start_time = time.time()


    avg_queue = []

    # 开始训练
    for episode in range(FLAGS.num_episodes):

        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        trajectories = []

        while not done:
            # 计算当前状态
            state = env.compute_state()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # 获取动作概率分布
            action_probs = agent.policy_net(state_tensor)
            action, log_prob = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 检查动作和奖励是否有效
            if info['do_action'] is not None and reward is not None:
                # 使用实际执行的动作计算 log_prob
                do_action = torch.tensor(info['do_action'], device=device)
                actual_log_prob = Categorical(action_probs).log_prob(do_action)


                trajectories.append((
                    state,
                    info['do_action'],
                    actual_log_prob.detach(),
                    reward,
                    next_state,
                    done
                ))


                del do_action


            del state_tensor, action_probs
            torch.cuda.empty_cache()


        if FLAGS.mode == 'train':
            agent.update(trajectories)


        env.close()


        print(f"Episode {episode} completed.")

        current_date = str(datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')


        if not os.path.exists('weights'):
            os.makedirs('weights')


        if FLAGS.mode == 'train' and episode % 25 == 0:
            torch.save(agent.policy_net.state_dict(), f'weights/ppo_weights_{current_date}_{episode}.pth')


        avg_queue.append(env.avg_queue[-1])
        print(f'Episode: {episode}, Average Queue: {env.avg_queue[-1]}')


        current_time = time.time()
        print(f'Time Elapsed: {current_time - start_time} seconds')


        if FLAGS.mode == 'train' and episode % 25 == 0:
            plot_average_queue(avg_queue, episode, current_date)
            print(f'Average Queue History: {avg_queue}')

        if FLAGS.mode == 'train' and episode % 25 == 0:
            with open("record/avg_queue.txt", "w") as file:
                file.write(','.join(map(str, avg_queue)))


if __name__ == '__main__':
    app.run(main)