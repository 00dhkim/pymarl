'''
# 코드 해석
sc2 환경만 여러 개 만들고, 모델 (즉, mac)은 하나.
(env_worker로 구현된) sc2에서 obs, reward 등등을 반환하면,
runner는 각 sc2 환경들의 최적 액션들을 선택해서 쭉 뿌려줌.

# SISR 구현
처음 시작하면 random worker와 origin worker를 각각 생성한 다음,
확률적으로 필요할 때에는 random worker를 활성화시키고,
그렇지 않을 때에는 origin worker만 활성화한다면 굳이 새롭게 _launch할 필요 없겠다!

# 구현 노트
일단 확률적으로 랜덤 스폰되도록 구현은 했음

[v] 구현한 코드에 컴파일 에러는 없는지, 한번에 하나의 worker만 실행되는지
[v] 각 worker마다 랜덤 여/부 맵 적용하기
[v] 각 worker가 활성화되었을 때 올바른지 (랜덤일때 진짜 랜덤인지, 원본일때 진짜 원본인지) 테스트 코드 작성 (유닛들 초기 위치 로깅)

'''

import copy
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class SISRRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        assert self.args.batch_size_run == 1, "batch_size_run must be 1 with sisr runner"
        self.batch_size = self.args.batch_size_run
        self.worker_size = 2

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.worker_size)])
        env_fn = env_REGISTRY[self.args.env]
        # self.ps[0] 는 origin worker
        # self.ps[1] 는 random worker
        # self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
        #                     for worker_conn in self.worker_conns]
        env_args_random_map = copy.copy(self.args.env_args)
        env_args_random_map['map_name'] += '_random'
        
        ps_origin = Process(target=env_worker, args=(self.worker_conns[0], CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
        ps_random = Process(target=env_worker, args=(self.worker_conns[1], CloudpickleWrapper(partial(env_fn, **env_args_random_map))))
        self.ps = [ps_origin, ps_random]
        
        self.random_worker_runned = True

        for p in self.ps:
            p.daemon = True
            p.start()
        print('origin worker', ps_origin.pid)
        print('random worker', ps_random.pid)
        
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        for parent_conn in self.parent_conns:
            data = parent_conn.recv() # clear buffer

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, random_spawn):
        
        if not random_spawn:
            parent_conn = self.parent_conns[0]
        else:
            parent_conn = self.parent_conns[1]
        
        self.batch = self.new_batch()

        # Reset the envs
        parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        data = parent_conn.recv()
        pre_transition_data["state"].append(data["state"])
        pre_transition_data["avail_actions"].append(data["avail_actions"])
        pre_transition_data["obs"].append(data["obs"])
        
        parent_conn.send(("test_random_spawn", None))

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, random_spawn=False):
        self.reset(random_spawn=random_spawn)

        if not random_spawn:
            parent_conn = self.parent_conns[0] # origin worker
            other_parent_conn = self.parent_conns[1]
        else:
            parent_conn = self.parent_conns[1] # random worker
            other_parent_conn = self.parent_conns[0]
        idx = 0
        
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0

            if idx in envs_not_terminated: # We produced actions for this env
                if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                    parent_conn.send(("step", cpu_actions[action_idx]))
                action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            if not terminated[idx]:
                data = parent_conn.recv()
                # Remaining data for this current timestep
                post_transition_data["reward"].append((data["reward"],))

                episode_returns[idx] += data["reward"]
                episode_lengths[idx] += 1
                if not test_mode:
                    self.env_steps_this_run += 1

                env_terminated = False
                if data["terminated"]:
                    final_env_infos.append(data["info"])
                if data["terminated"] and not data["info"].get("episode_limit", False):
                    env_terminated = True
                terminated[idx] = data["terminated"]
                post_transition_data["terminated"].append((env_terminated,))

                # Data for the next timestep needed to select an action
                pre_transition_data["state"].append(data["state"])
                pre_transition_data["avail_actions"].append(data["avail_actions"])
                pre_transition_data["obs"].append(data["obs"])
            
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        # 현재 동작중인 worker와 쉬고 있는 worker의 _total_steps, _episode_count 싱크 맞춤
        # 더 싱크 맞춰야하는 항목 없는가? 없겠지...
        parent_conn.send(("get_timesteps", None))
        total_steps, episode_count = parent_conn.recv()
        
        # other_parent_conn.send(("get_timesteps", None)) # debug
        # other_total_steps, other_episode_count = other_parent_conn.recv() # debug
        # if other_total_steps + episode_lengths[idx] != total_steps: # debug
        #     print('other_total_steps', other_total_steps)
        #     print('episode_lengths[idx]', episode_lengths[idx])
        #     print('total_steps', total_steps)
        #     assert False
        
        other_parent_conn.send(("set_timesteps", (total_steps, episode_count)))
        
        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        parent_conn.send(("get_stats",None))

        env_stats = []
        env_stat = parent_conn.recv()
        env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        ###########
        elif cmd == "get_timesteps":
            remote.send(env.get_timesteps())
        elif cmd == "set_timesteps":
            total_steps, episode_count = data
            env.set_timesteps(total_steps, episode_count)
        elif cmd == "test_random_spawn":
            env.test_random_spawn()
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

