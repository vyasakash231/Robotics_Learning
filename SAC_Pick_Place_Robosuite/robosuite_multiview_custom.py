import sys
import numpy as np
import pygame
from collections import OrderedDict

from robosuite.wrappers import VisualizationWrapper

class MultiViewWrapper(VisualizationWrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        pygame.init()

        self.env_name = env_name
        
        # Set up display parameters
        self.screen_width = 1024
        self.screen_height = 768
        self.view_width = self.screen_width // 2
        self.view_height = self.screen_height // 2
        
        # Initialize pygame display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robosuite Multi-View")

        # Map the available cameras to our desired views
        if self.env_name == "Lift":
            self.camera_mapping = {
                "view_1": "agentview",
                "view_2": "birdview",
                "view_3": "sideview",
                "view_4": "robot0_robotview"}
            
        if self.env_name in ["PickPlace", "PickPlaceSingle", "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan"]:
            self.camera_mapping = {
                "view_1": "agentview",
                "view_2": "birdview",
                "view_3": "frontview",
                "view_4": "robot0_robotview"}

        # Initialize camera adjustment mode
        self.adjust_mode = False
        self.current_camera = None
        
        # Camera adjustment parameters
        self.distance_step = 0.1
        self.azimuth_step = 5
        self.elevation_step = 5
        self.lookat_step = 0.1

        # Store camera parameters
        self.camera_params = {name: {
            'distance': 2.0,
            'azimuth': 90.0,
            'elevation': -45.0,
            'lookat': np.array([0.0, 0.0, 1.0])
        } for name in self.camera_mapping.values()}

        # Save initial camera positions and orientations
        self._save_initial_camera_states()

    def _save_initial_camera_states(self):
        """Save initial camera states for reference"""
        self.initial_camera_states = {}
        # print(self.camera_mapping.values())
        for camera_name in self.camera_mapping.values():
            camera_id = self.env.sim.model.camera_name2id(camera_name)
            self.initial_camera_states[camera_name] = {
                'pos': self.env.sim.model.cam_pos[camera_id].copy(),
                'quat': self.env.sim.model.cam_quat[camera_id].copy()
            }

    def update_camera_view(self, camera_name):
        """Update camera view based on spherical coordinates"""
        if camera_name not in self.camera_params:
            return

        params = self.camera_params[camera_name]
        distance = params['distance']
        azimuth = np.deg2rad(params['azimuth'])
        elevation = np.deg2rad(params['elevation'])
        lookat = params['lookat']

        # Convert spherical to cartesian coordinates
        x = distance * np.cos(elevation) * np.cos(azimuth)
        y = distance * np.cos(elevation) * np.sin(azimuth)
        z = distance * np.sin(elevation)

        # Calculate camera position
        cam_pos = lookat + np.array([x, y, z])
        
        # Get camera ID and update position
        camera_id = self.env.sim.model.camera_name2id(camera_name)
        self.env.sim.model.cam_pos[camera_id] = cam_pos

        # Calculate camera orientation
        forward = lookat - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Create rotation matrix
        rot_mat = np.vstack([right, up, -forward]).T
        
        # Convert to quaternion
        quat = self.rot2quat(rot_mat)
        self.env.sim.model.cam_quat[camera_id] = quat

        # Forward the simulation to update camera
        self.env.sim.forward()

    def rot2quat(self, rot_mat):
        """Convert rotation matrix to quaternion"""
        tr = np.trace(rot_mat)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (rot_mat[2, 1] - rot_mat[1, 2]) / S
            qy = (rot_mat[0, 2] - rot_mat[2, 0]) / S
            qz = (rot_mat[1, 0] - rot_mat[0, 1]) / S
        elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            S = np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
            qw = (rot_mat[2, 1] - rot_mat[1, 2]) / S
            qx = 0.25 * S
            qy = (rot_mat[0, 1] + rot_mat[1, 0]) / S
            qz = (rot_mat[0, 2] + rot_mat[2, 0]) / S
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            S = np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
            qw = (rot_mat[0, 2] - rot_mat[2, 0]) / S
            qx = (rot_mat[0, 1] + rot_mat[1, 0]) / S
            qy = 0.25 * S
            qz = (rot_mat[1, 2] + rot_mat[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
            qw = (rot_mat[1, 0] - rot_mat[0, 1]) / S
            qx = (rot_mat[0, 2] + rot_mat[2, 0]) / S
            qy = (rot_mat[1, 2] + rot_mat[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    def handle_keyboard_input(self):
        """Handle keyboard input for camera adjustment"""
        keys = pygame.key.get_pressed()
        
        if not self.adjust_mode:
            if keys[pygame.K_TAB]:
                self.adjust_mode = True
                self.current_camera = "agentview"
                print(f"Camera adjustment mode: {self.current_camera}")
            return

        # Switch between cameras
        if self.env_name == "Lift":
            if keys[pygame.K_1]:
                self.current_camera = "agentview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_2]:
                self.current_camera = "birdview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_3]:
                self.current_camera = "sideview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_4]:
                self.current_camera = "robot0_robotview"
                print(f"Adjusting camera: {self.current_camera}")
                
        if self.env_name in ["PickPlace", "PickPlaceSingle", "PickPlaceMilk", "PickPlaceBread", "PickPlaceCereal", "PickPlaceCan"]:
            if keys[pygame.K_1]:
                self.current_camera = "agentview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_2]:
                self.current_camera = "birdview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_3]:
                self.current_camera = "frontview"
                print(f"Adjusting camera: {self.current_camera}")
            elif keys[pygame.K_4]:
                self.current_camera = "robot0_robotview"
                print(f"Adjusting camera: {self.current_camera}")

        if self.current_camera:
            params = self.camera_params[self.current_camera]
            modified = False
            
            # Distance (zoom)
            if keys[pygame.K_w]:
                params['distance'] = max(0.1, params['distance'] - self.distance_step)
                modified = True
            if keys[pygame.K_s]:
                params['distance'] += self.distance_step
                modified = True

            # Azimuth (horizontal rotation)
            if keys[pygame.K_a]:
                params['azimuth'] = (params['azimuth'] - self.azimuth_step) % 360
                modified = True
            if keys[pygame.K_d]:
                params['azimuth'] = (params['azimuth'] + self.azimuth_step) % 360
                modified = True

            # Elevation (vertical rotation)
            if keys[pygame.K_q]:
                params['elevation'] = min(89, params['elevation'] + self.elevation_step)
                modified = True
            if keys[pygame.K_e]:
                params['elevation'] = max(-89, params['elevation'] - self.elevation_step)
                modified = True

            # Move look-at point
            # if keys[pygame.K_LEFT]:
            #     params['lookat'][0] -= self.lookat_step
            #     modified = True
            # if keys[pygame.K_RIGHT]:
            #     params['lookat'][0] += self.lookat_step
            #     modified = True
            # if keys[pygame.K_UP]:
            #     params['lookat'][1] += self.lookat_step
            #     modified = True
            # if keys[pygame.K_DOWN]:
            #     params['lookat'][1] -= self.lookat_step
            #     modified = True
            if keys[pygame.K_PAGEUP]:
                params['lookat'][2] += self.lookat_step
                modified = True
            if keys[pygame.K_PAGEDOWN]:
                params['lookat'][2] -= self.lookat_step
                modified = True

            # Reset camera to initial position
            if keys[pygame.K_r]:
                camera_id = self.env.sim.model.camera_name2id(self.current_camera)
                self.env.sim.model.cam_pos[camera_id] = self.initial_camera_states[self.current_camera]['pos']
                self.env.sim.model.cam_quat[camera_id] = self.initial_camera_states[self.current_camera]['quat']
                self.env.sim.forward()
                modified = True
                print(f"Reset {self.current_camera} to initial position")

            # Update the camera view if any parameter was modified
            if modified:
                self.update_camera_view(self.current_camera)

            # Exit adjustment mode
            if keys[pygame.K_ESCAPE]:
                self.adjust_mode = False
                print("Exiting camera adjustment mode")
                print(f"Final camera parameters for {self.current_camera}:")
                print(f"Distance: {params['distance']}")
                print(f"Azimuth: {params['azimuth']}")
                print(f"Elevation: {params['elevation']}")
                print(f"Look-at point: {params['lookat']}")

    def render(self):
        """Render multiple views using existing cameras"""
        # Handle camera adjustments
        self.handle_keyboard_input()

        views = OrderedDict()
        
        # Get images from different cameras
        for view_name, camera_name in self.camera_mapping.items():
            img = self.env.sim.render(
                camera_name=camera_name,
                width=self.view_width,
                height=self.view_height,
                depth=False
            )
            views[view_name] = img

        # Display positions for each view
        positions = {
            "view_1": (0, 0),
            "view_2": (self.view_width, 0),
            "view_3": (0, self.view_height),
            "view_4": (self.view_width, self.view_height)}
            
        # Convert and display images
        for view_name, img in views.items():
            img = img[::-1, :, ::-1]
            surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            self.screen.blit(surface, positions[view_name])

        # Add labels and show current camera parameters if in adjust mode
        font = pygame.font.Font(None, 36)

        labels = {
            "view_1": "Window (1)",
            "view_2": "Window (2)",
            "view_3": "Window (3)",
            "view_4": "Window (4)"}
            
        if self.adjust_mode and self.current_camera:
            for view_name, camera_name in self.camera_mapping.items():
                if camera_name == self.current_camera:
                    params = self.camera_params[camera_name]
                    labels[view_name] += f"\nD:{params['distance']:.1f} A:{params['azimuth']:.0f} E:{params['elevation']:.0f}"

        for view_name, pos in positions.items():
            if (self.adjust_mode and self.camera_mapping[view_name] == self.current_camera):
                label_color = (255, 255, 0)
            else:
                label_color = (255, 255, 255)
            
            label_lines = labels[view_name].split('\n')
            for i, line in enumerate(label_lines):
                label_surface = font.render(line, True, label_color)
                self.screen.blit(label_surface, (pos[0] + 10, pos[1] + 10 + i * 20))

        pygame.display.flip()