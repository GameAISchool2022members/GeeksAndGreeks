"""
Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates
You can visualize experiment results in ~/ray_results using TensorBoard.
Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import math

import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune import Callback, CLIReporter, ExperimentAnalysis, Stopper

from env import Env
from model import CustomConvModel

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=math.inf, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=math.inf, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=math.inf, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="0",
)
parser.add_argument(
    "--infer",
    action="store_true",
)


class CustomPPOTrainer(PPOTrainer):
    log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # wandb.init(**self.config['wandb'])
        # self.checkpoint_path_file = kwargs['config']['checkpoint_path_file']
        # self.ctrl_metrics = self.config['env_config']['conditionals']
        # cbs = self.workers.foreach_env(lambda env: env.unwrapped.cond_bounds)
        # cbs = [cb for worker_cbs in cbs for cb in worker_cbs if cb is not None]
        # cond_bounds = cbs[0]
        # self.checkpoint_path_file = checkpoint_path_file

    def setup(self, config):
        ret = super().setup(config)
        n_params = 0
        param_dict = self.get_weights()['default_policy']

        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'default_policy has {n_params} parameters.')
        # print('model overview: \n', self.get_policy('default_policy').model)
        return ret

    @classmethod
    def get_default_config(cls):
        def_cfg = PPOTrainer.get_default_config()
        def_cfg.update({
            # 'wandb': {
            #     'project': 'PCGRL',
            #     'name': 'default_name',
            #     'id': 'default_id',
            # },
            "exp_id": 0,
        })
        return def_cfg

    # def save(self, *args, **kwargs):
    #     ckp_path = super().save(*args, **kwargs)
    #     with open(self.checkpoint_path_file, 'w') as f:
    #         f.write(ckp_path)
    #     return ckp_path

    # @wandb_mixin
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        log_result = {k: v for k, v in result.items() if k in self.log_keys}
        log_result['info: learner:'] = result['info']['learner']

        # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
        result['fps'] = result['num_agent_steps_trained'] / result['time_this_iter_s']
        return result


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)
    local_dir = "./runs"
    # trainer_name TorchCustomModel= "dgPPO"
    trainer_name = "dgPPO"
    tune.register_trainable(trainer_name, CustomPPOTrainer)
    exp_name = f"{trainer_name}_{args.exp_name}"

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: Env(config))
    ModelCatalog.register_custom_model(
        # "my_model", TorchCustomModel
        "my_model", CustomConvModel
    )

    config = {
        "env": Env,  # or "corridor" if registered above
        "env_config": {
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": 5e-6,
        "num_envs_per_worker": 20 if not args.infer else 1,
        "num_workers": 6 if not args.infer else 0,  # parallelism
        "framework": args.framework,
        "render_env": args.infer,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.infer:
        # config['lr'] = 0.0  # dummy to allow us to initialize the trainer without issue
        trainer = CustomPPOTrainer(config=config)
        analysis = ExperimentAnalysis(os.path.join(local_dir, exp_name))
        ckp_paths = [analysis.get_trial_checkpoints_paths(analysis.trials[i]) for i in range(len(analysis.trials))]
        assert np.all([len(paths) == 1 for paths in ckp_paths]), f"Expected 1 checkpoint per trial, got {[len(paths) for paths in ckp_paths]}."
        ckp_paths = [p for paths in ckp_paths for p in paths]
        for ckp_path in ckp_paths:
            if args.infer:
                trainer.restore(ckp_path[0])
                for i in range(10):
                    print(f'eval {i}')
                    trainer.evaluate()
            # elif args.resume_sequential:
                # analysis = launch_analysis()

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=Env)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(run_or_experiment=trainer_name, config=config, stop=stop, checkpoint_freq=1, keep_checkpoints_num=1, 
            name=exp_name, local_dir=local_dir)

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()