import time
from piper_control import piper_connect
from piper_control import piper_control
from piper_control import piper_init
from piper_control import piper_interface
from gello.robots.robot import Robot

from typing import Dict
import numpy as np

MAX_OPEN = 0.09
WRIST_OFFSET = -0.74

class PiperRobot(Robot):
    """A class representing a piper robot."""

    def __init__(self):
        self.connect()
        self.robot = piper_interface.PiperInterface(can_port = self.can_id)
        time.sleep(1)
        self.reset()
        time.sleep(3)
    
    def connect(self):
        ports = piper_connect.find_ports()
        print(ports)

        piper_connect.activate()
        self.can_id = piper_connect.active_ports()[0]

    def reset(self):

        ##### THIS WILL FALL DOWN ######

        self.robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)
        piper_init.reset_arm(
        self.robot,
        arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
        move_mode=piper_interface.MoveMode.JOINT,
        )
        piper_init.reset_gripper(self.robot)
        self.command_joint_state(np.zeros(self.num_dofs()))
        

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 7

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the robot.

        Returns:
            T: The current state of the robot.
        """
        robot_joints = self.robot.get_joint_positions()
        gripper_pos = self.robot.get_gripper_state()[0]
        pos = np.append(robot_joints, gripper_pos)
        pos[5] -= WRIST_OFFSET
        return pos

    def command_joint_state(self, joint_state) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        joint_state = joint_state[:self.num_dofs() - 1]
        print(joint_state)
        self.send_arm_to_pos(joint_state.tolist())

    def send_arm_to_pos(self, joints):
        joints = list(joints)
        joints[5] += WRIST_OFFSET
        self.robot.command_joint_positions(joints)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }
    
    def rest(self):
        pos = piper_control.ArmOrientations.upright.rest_position
        pos = list(pos)
        pos[4] = 0.4
        self.send_arm_to_pos(pos)


    def close(self):
        self.rest()

        # breakpoint()    
        time.sleep(4)
        piper_init.disable_arm(self.robot, timeout_seconds=10)

    def zero(self):
        self.robot.set_joint_zero_positions([5])
    
    #0.00000000e+00  0.00000000e+00  0.00000000e+00  2.00363795e-02
  #0.00000000e+00 -7.40194123e-01 -4.20000000e-04


def main():
    robot = PiperRobot()
    current_joints = robot.get_joint_state()
    print(current_joints)
    time.sleep(3)
    robot.close()

if __name__ == "__main__":
    main()
