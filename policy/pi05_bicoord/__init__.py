"""BiCoord evaluation adapter using the trained all-task pi05 EMA policy."""
import numpy as np
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import bicoord
from pathlib import Path


class Model:
    def __init__(self,checkpoint=None):
        cfg=bicoord.get_config()
        self.pi0_step=cfg.model.action_horizon
        path=Path(checkpoint) if checkpoint else bicoord.ROOT/"checkpoints/50000"
        kwargs={}
        if path.resolve()==(bicoord.WS/"ckpts/pi05_base").resolve():
            # Pre-training smoke test uses the same BiCoord input normalization.
            kwargs["norm_stats"]=normalize.load(cfg.assets_dirs/bicoord.ASSET_ID)
        self.policy=policy_config.create_trained_policy(cfg,path,
                         robotwin_repo_id=bicoord.ASSET_ID,**kwargs)


def get_model(usr_args):
    return Model(usr_args.get("checkpoint_dir"))


def reset_model(model):
    pass


def eval(TASK_ENV,model,observation):
    # Same physical input mapping and receding horizon50 as official pi0.
    images={"cam_high":observation["observation"]["head_camera"]["rgb"],
            "cam_left_wrist":observation["observation"]["left_camera"]["rgb"],
            "cam_right_wrist":observation["observation"]["right_camera"]["rgb"]}
    inputs={"state":observation["joint_action"]["vector"],
            "images":{name:np.transpose(image,(2,0,1)) for name,image in images.items()},
            "prompt":TASK_ENV.get_instruction()}
    actions=model.policy.infer(inputs)["actions"]
    for action in actions:
        TASK_ENV.take_action(action)
        # Preserve official adapter's observation refresh at every action.
        observation=TASK_ENV.get_obs()
