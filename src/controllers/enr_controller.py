from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class ENRMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states_list = [None] * args.mc_approx
        self.clean_hidden_states = None

    def select_actions(self, clean_flag, ep_batches, t_ep, t_env, bs=slice(None), test_mode=False): # runner에서만 호출됨
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batches[0]["avail_actions"][:, t_ep] # avail_actions는 모든 batch가 동일함
        agent_outputs = self.forward(clean_flag, ep_batches, t_ep, test_mode=test_mode)
        # 리턴하는 agent_outputs 는 모든 batch를 aggregate한 결과임.
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, clean_flag, ep_batches, t, test_mode=False):
        # 학습중이라면 clean_flag == True
        
        for idx, ep_batch in enumerate(ep_batches):
            pass###################################
        
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # fm_loss를 계산하기 위해 clean_path, random_path 각각의 hidden feature를 구할 필요가 있음.
        # 그렇기에 둘 다 실행하는 것.
        clean_agent_outs, self.clean_hidden_states = self.agent(True, agent_inputs, self.clean_hidden_states, idx_mc=0)
        
        agent_outs = []
        hidden_states_list = []
        for idx_mc in range(self.args.mc_approx):
            curr_agent_outs, curr_hidden_states = self.agent(False, agent_inputs, self.hidden_states_list[idx_mc], idx_mc=idx_mc)
            agent_outs.append(curr_agent_outs)
            hidden_states_list.append(curr_hidden_states)
        # 적당히 평균하여 aggregate
        agent_outs = th.stack(agent_outs)
        agent_outs = agent_outs.mean(dim=0)
        self.hidden_states_list = hidden_states_list
        assert agent_outs.shape == clean_agent_outs.shape
        assert self.hidden_states_list[0].shape == self.clean_hidden_states.shape
        
        if clean_flag:
            agent_outs = clean_agent_outs
        else:
            agent_outs = agent_outs

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size, seed):
        self.hidden_states_list = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.clean_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.agent.init_random_layer(seed)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def forward_random_layer(self):
        raise NotImplementedError