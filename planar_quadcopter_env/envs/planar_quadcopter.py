import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from math import remainder
from scipy.integrate import solve_ivp


class PlanarQuadCopterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    dt = 0.1
    MASS = 0.486     # kg
    INERTIA = 0.00383    # kg·m^2
    CENTER_TO_PROP = 0.25     # m
    GRAVITY = 9.81     # m/s^2
    # alpha= 1/5 * np.arctanh(1/2)
    alpha= 1
    beta = GRAVITY


    def __init__(self, box_size=10, render_mode="rgb_array"):
        self.size = None  # The size of the square grid
        self.HEIGHT_WIDTH=box_size
        self.window_size = 512  # The size of the PyGame window

        # Observation is [x,y,theta, vx, vy, vtheta]
        # Max positional velocities is freefall from top to bottom:
        # sqrt(2g(ymax-ymin))
        # Max angular velocity is 1 revolution maximum 2sqrt(pi*m*g*r/I)
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([-self.HEIGHT_WIDTH/2,-self.HEIGHT_WIDTH/2,-np.pi,-50,-50,2*-65]),
            high=np.array([self.HEIGHT_WIDTH/2,self.HEIGHT_WIDTH/2,np.pi,50,50,2*65]),
            dtype=np.float64
        )
        # Action space of two controls
        self.action_space: spaces.Box = spaces.Box(low=np.array([-self.beta/2,-self.beta/2]),
                                       high=np.array([self.beta/2,self.beta/2]),
                                       dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_state

    def _get_info(self):
        return {
            "distance": None
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # default_options = {
        #     "low": np.array([15, 15, -np.pi/4, -2, -2, -4]),
        #     "high": np.array([25, 25, np.pi/4, 2, 2, 4])
        # }
        default_options = {
            "low": np.array([-2/5 * self.HEIGHT_WIDTH, -2/5*self.HEIGHT_WIDTH, -np.pi/4, -2, -2, -4]),
            "high": np.array([2/5*self.HEIGHT_WIDTH, 2/5*self.HEIGHT_WIDTH, np.pi/4, 2, 2, 4]),
            "saturate_controls": True
        }

        if options is not None:
            if "low" in options:
                assert options["low"].shape == default_options["low"].shape
                default_options["low"] = options["low"]
            if "high" in options:
                assert options["high"].shape == default_options["high"].shape
                default_options["high"] = options["high"]
            if "saturate_controls" in options:
                assert isinstance(options["saturate_controls"], bool)
                default_options["saturate_controls"] = options["saturate_controls"]
        
        options = default_options

        # Choose the agent's location uniformly at random
        self._agent_state = self.np_random.uniform(options["low"],options["high"])
        self._action = np.array([0,0])
        self._options = options

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        self._action = np.clip(action, self.action_space.low, self.action_space.high)/self.action_space.high
        # print(f"{self._action=}")
        u1,u2 = action

        control_u1 = np.tanh(self.alpha*u1) if self._options["saturate_controls"] else u1
        control_u2 = np.tanh(self.alpha*u2) if self._options["saturate_controls"] else u2
        def gradient(state):
            _, _, theta, vx, vy, vtheta = state
            ddx = -self.beta*(control_u1 + control_u2)*np.sin(theta)/self.MASS
            ddy = self.beta*(control_u1 + control_u2)*np.cos(theta)/self.MASS - self.GRAVITY
            ddz = self.CENTER_TO_PROP/self.INERTIA*self.beta*(control_u1 - control_u2)

            return np.array([vx,vy,vtheta, ddx,ddy,ddz])

        # print(gradient(self._agent_state))

        new_state = solve_ivp(
            lambda _,y: gradient(y),
            t_span=[0.0, self.dt],
            y0 = self._agent_state,
            t_eval=[self.dt]
        ).y.flatten()
        # print(gradient(new_state))

        new_state[2] = remainder(new_state[2], 2*np.pi) # Keep angle [-pi,pi]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_state = np.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )
        # An episode is done iff the agent has reached the target
        terminated = np.any(
            (self._agent_state[:2] == self.observation_space.low[:2]) |
            (self._agent_state[:2] == self.observation_space.high[:2])
        )

        # terminated = False
        # reward = 1 if terminated else 0  # Binary sparse rewards
        reward = 1 if not terminated else -500
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        pix_per_meter = self.window_size / self.HEIGHT_WIDTH

        # Convert from -50,50 to 0,100 coordinates
        x,y = self._agent_state[:2]
        pygame_agent_state = np.array([x,-y]) + self.HEIGHT_WIDTH/2

        # First we draw body, rotated by theta
        body_height = 0.3*self.CENTER_TO_PROP*2*5 * pix_per_meter
        body_width = self.CENTER_TO_PROP * 2*4*5 * pix_per_meter
        theta = self._agent_state[2]

        # Create a surface for the body rectangle
        body_surf = pygame.Surface((body_width, body_height + body_width*2), pygame.SRCALPHA)
        # Draw the body
        pygame.draw.rect(
            body_surf,
            (0, 0, 0),
            pygame.Rect(0, body_width, body_width, body_height)
        )
        # Draw a top indicator
        pygame.draw.rect(
            body_surf,
            (0, 0, 0),
            pygame.Rect(body_width/2 - body_height/2, body_width - body_height, body_height, body_height)
        )
        # Draw arrows indicating control strength
        pygame.draw.line(
            body_surf,
            (255,0,0),
            (body_height/2, body_height/2 + body_width),
            (body_height/2, body_height/2 + body_width + self._action[0]*body_width),
            int(body_height)
        )
        pygame.draw.line(
            body_surf,
            (255,0,0),
            (body_width - body_height/2, body_height/2 + body_width),
            (body_width - body_height/2, body_height/2 + body_width + self._action[1]*body_width),
            int(body_height)
        )

        # Rotate the surface by theta (convert to degrees)
        rotated_body = pygame.transform.rotate(body_surf, np.degrees(theta))

        # Get the new rect and set its center to the agent's position
        rotated_rect = rotated_body.get_rect(center=(pix_per_meter * pygame_agent_state))

        # Blit the rotated body onto the canvas
        canvas.blit(rotated_body, rotated_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
