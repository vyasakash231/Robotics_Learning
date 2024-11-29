import sys
import numpy as np
import pygame

class Controller:
    def __init__(self, control_mode="OSC_POSE"):
        """Initialize pygame and the game controller."""
        pygame.init()
        pygame.joystick.init()

        self.control_mode = control_mode
        
        if pygame.joystick.get_count() > 0:  # check no of controller connected
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            self.gripper_close = None
        
            # Control scaling factors
            self.translation_scale = 0.2  # Scale factor for position control
            self.rotation_scale = 0.3     # Scale factor for rotation control
            
            print(f"\nInitialized {self.controller.get_name()}")
            print(f"Number of axes: {self.controller.get_numaxes()}")
            print(f"Number of buttons: {self.controller.get_numbuttons()}")
            print("\nControl Mapping (OSC_POSE Mode):")
            print("--------------------------------")
            print("Left Stick X: Move end-effector left/right (dx)")
            print("Left Stick Y: Move end-effector forward/backward (dy)")
            print("Right Stick Y: Move end-effector up/down (dz)")
            print("Right Stick X: Roll (droll)")
            print("RT + RB/no RB: Pitch up/down (dpitch)")
            print("LT + LB/no LB: Yaw left/right (dyaw)")
            print("X Button: Open gripper")
            print("B Button: Close gripper")
        else:
            print("No game controller detected")

    def get_actions(self):
        """Get controller inputs and convert to robot actions based on control mode."""
        if self.control_mode == "OSC_POSE":
            return self._get_osc_pose_actions()
        else:
            return self._get_joint_velocity_actions()

    def _get_osc_pose_actions(self):
        """Get controller inputs mapped to OSC_POSE control."""
        actions = np.zeros(7)  # [dx, dy, dz, droll, dpitch, dyaw, gripper]

        # Handle translation (Left stick XY, Right stick Y for Z)
        # Scale inputs for smoother control
        actions[0] = self.controller.get_axis(0) * self.translation_scale  # dx (left/right)
        actions[1] = -self.controller.get_axis(1) * self.translation_scale  # dy (forward/backward)
        actions[2] = -self.controller.get_axis(3) * self.translation_scale  # dz (up/down)

        # Handle rotation
        # Roll with right stick X
        actions[3] = self.controller.get_axis(2) * self.rotation_scale  # droll

        # Pitch with right trigger (RT)
        rt_value = self.controller.get_axis(4)
        rt_polarity = 1 if self.controller.get_button(7) else -1  # RB for polarity
        if rt_value > -0.9:  # Only when trigger is pressed
            actions[4] = rt_polarity * (rt_value + 1) / 2 * self.rotation_scale  # dpitch

        # Yaw with left trigger (LT)
        lt_value = self.controller.get_axis(5)
        lt_polarity = 1 if self.controller.get_button(6) else -1  # LB for polarity
        if lt_value > -0.9:  # Only when trigger is pressed
            actions[5] = lt_polarity * (lt_value + 1) / 2 * self.rotation_scale  # dyaw

        # Apply deadzone to stick inputs
        deadzone = 0.1
        for i in range(4):  # Apply to stick axes only
            if abs(actions[i]) < deadzone:
                actions[i] = 0.0

        # Handle gripper
        if self.controller.get_button(3):  # X button
            self.gripper_close = False
            actions[6] = 1.0  # Open
        elif self.controller.get_button(1):  # B button
            self.gripper_close = True
            actions[6] = -1.0  # Close

        # Only return actions if there's significant input
        if np.all(actions[:6] == 0) and actions[6] == 0:
            return None

        # Print current actions for debugging
        print("\rActions - ", end="")
        labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'grip']
        for i, (val, label) in enumerate(zip(actions, labels)):
            if val != 0:
                print(f"{label}: {val:.2f} ", end="")
        print("", end="", flush=True)
        return actions

    def _get_joint_velocity_actions(self):
        """Original joint velocity control implementation."""
        actions = np.zeros(7)

        # Map joysticks (first 4 DOF)
        for i in range(4):
            value = self.controller.get_axis(i)
            actions[i] = value

        # Handle triggers with polarity control
        rt_value = self.controller.get_axis(4)
        rt_polarity = 1 if self.controller.get_button(7) else -1
        if rt_value > -0.9:
            actions[4] = rt_polarity * (rt_value + 1) / 2

        lt_value = self.controller.get_axis(5)
        lt_polarity = 1 if self.controller.get_button(6) else -1
        if lt_value > -0.9:
            actions[5] = lt_polarity * (lt_value + 1) / 2

        # Apply deadzone
        deadzone = 0.1
        for i in range(4):
            if abs(actions[i]) < deadzone:
                actions[i] = 0.0

        # Handle gripper
        if self.controller.get_button(3):  # X button
            self.gripper_close = False
            actions[6] = 1.0
        elif self.controller.get_button(1):  # B button
            self.gripper_close = True
            actions[6] = -1.0

        if np.all(actions[:6] == 0) and actions[6] == 0:
            return None
        return actions