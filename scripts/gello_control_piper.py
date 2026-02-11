from piper_control import piper_connect
from piper_control import piper_control
from piper_control import piper_init
from piper_control import piper_interface
from gello.env import RobotEnv
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.robot_node import ZMQClientRobot

import time
import glob
import tyro
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

class GELLOPiper:
  def __init__(self, Args):
    self.args = Args
    self.init_gello()
    self.connect()
    self.robot = piper_interface.PiperInterface(can_port=self.can_id)
    self.reset()

  def connect(self):
    ports = piper_connect.find_ports()
    print(ports)

    piper_connect.activate()
    self.can_id = piper_connect.active_ports()[0]

  def reset(self):

    ##### THIS WILL FALL DOWN ######

    piper_init.reset_arm(
      self.robot,
      arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
      move_mode=piper_interface.MoveMode.JOINT,
    )
    piper_init.reset_gripper(self.robot)

  def init_gello(self):
    self.robot_client = ZMQClientRobot(port=self.args.robot_port, host=self.args.hostname)
    self.env = RobotEnv(self.robot_client, control_rate_hz=self.args.hz, camera_dict={})
    
    gello_port = self.args.gello_port
    
    usb_ports = glob.glob("/dev/serial/by-id/*")
    print(f"Found {len(usb_ports)} ports")
    if len(usb_ports) > 0:
        gello_port = usb_ports[0]
        print(f"using port {gello_port}")
    else:
        raise ValueError(
            "No gello port found, please specify one or plug in gello"
        )
    self.agent_cfg = {
        "_target_": "gello.agents.gello_agent.GelloAgent",
        "port": gello_port,
        "start_joints": self.args.start_joints,
    }
    if self.args.start_joints is None:
        reset_joints = np.deg2rad(
            [0, 0, 0, 0, 0, 0, 0]
        )  # Change this to your own reset joints
    else:
        reset_joints = np.array(self.args.start_joints)
    print("getting obs")
    curr_joints = self.env.get_obs()["joint_positions"]
    print(curr_joints)
    if reset_joints.shape == curr_joints.shape:
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            self.env.step(jnt)
            time.sleep(0.001)

    self.agent = instantiate_from_dict(self.agent_cfg)
    # going to start position
    print("Going to start position")
    start_pos = self.agent.act(self.env.get_obs())
    obs = self.env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = self.env.get_obs()
        command_joints = self.agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        self.env.step(current_joints + delta)

    self.obs = self.env.get_obs()



  def get_gello_joints(self):
    action = self.agent.act(self.obs)
    self.obs = self.env.step(action)

    return action

  def listen_to_gello(self):
    joint_angles = self.get_gello_joints()
    breakpoint()
    print(self.robot.get_joint_positions())
    self.robot.command_joint_positions(joint_angles)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args):

  piper = GELLOPiper(Args=args)
  while True:
    breakpoint()
    piper.listen_to_gello()
    time.sleep(0.01)

if __name__ == "__main__":
  main(tyro.cli(Args))
