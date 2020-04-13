from slm_lab.spec import GridSearch, LinearSchedule

spec = {
  "dqn_cartpole": {
    "agent": [{
      "name": "DQN",
      "algorithm": {
        "name": "DQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var": LinearSchedule(
            start_val=1.0,
            end_val=0.1,
            start_step=0,
            end_step=1000,
        ),
        "gamma": GridSearch([0.95, 0.99]),
        "clip_eps": LinearSchedule(
            start_val=0.01,
            end_val=0.001,
            start_step=100,
            end_step=5000,
        ),
        "training_batch_iter": 8,
        "training_iter": 4,
        "training_frequency": 4,
        "training_start_step": 32
      },
      "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": True
      },
      "net": {
        "type": "MLPNet",
        "hid_layers": [64],
        "hid_layers_activation": "selu",
        "clip_grad_val": 0.5,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": GridSearch([0.001, 0.01])
        },
        "lr_scheduler_spec": {
          "name": "StepLR",
          "step_size": 1000,
          "gamma": 0.9,
        },
        "update_type": "polyak",
        "update_frequency": 32,
        "polyak_coef": 0.1,
        "gpu": False
      }
    }],
    "env": [{
      "name": "CartPole-v0",
      "max_t": None,
      "max_frame": 10000
    }],
    "body": {
      "product": "outer",
      "num": 1
    },
    "meta": {
      "distributed": False,
      "eval_frequency": 2000,
      "max_trial": 4,
      "max_session": 2,
      "search": "RandomSearch",
    },
  }
}
