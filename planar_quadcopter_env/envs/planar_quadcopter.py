import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from math import remainder
import warnings


class PlanarQuadCopterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    dt = 0.1
    MASS = 1     # kg   Dylan's = 0.486
    INERTIA = 0.25    # kg·m^2  Dylan's = 0.00383
    CENTER_TO_PROP = 0.1     # m   Dylan's = 0.25
    GRAVITY = 9.8     # m/s^2   Dylan's = 9.81
    # alpha= 1/5 * np.arctanh(1/2)
    alpha = 1
    beta = 1 # Dylan's GRAVITY


    def __init__(self, box_size=25, render_mode="rgb_array", noisy=False, num_obs=5, obs_low=1, 
                 obs_high=3, goal_rad=6, goal_excl=15, obstacles=None, goal=None): # changed default box from 10 to 25
        # box_size: environment size of box_size x box_size
        # noisy: whether Gaussian noise is added to dynamics
        # num_obs: number of obstacles (to be uniformly sampled from state space), default is 5
        # obs_low: lower bound on obstacle radius, default is 1
        # obs_high: upper bound on bostacle radius, default is 3
        # goal_rad: radius for the goal region, default is 6
        # goal_excl: minimum distance that initial state must be from goal center, default is 15
        
        self.size = None  # The size of the square grid
        self.HEIGHT_WIDTH=box_size
        self.window_size = 512  # The size of the PyGame window
        self.noisy = noisy

        # Observation is [x,y,theta, vx, vy, vtheta]
        # Max positional velocities is freefall from top to bottom:
        # sqrt(2g(ymax-ymin))
        # Max angular velocity is 1 revolution maximum 2sqrt(pi*m*g*r/I)
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([-self.HEIGHT_WIDTH/2,-self.HEIGHT_WIDTH/2,-np.pi,-50,-50,2*-65]),
            high=np.array([self.HEIGHT_WIDTH/2,self.HEIGHT_WIDTH/2,np.pi,50,50,2*65]),
            dtype=np.float64
        )
        # Action space of two controls, uses thrust and torque
        # Action space for thrust and torque
        self.action_space: spaces.Box = spaces.Box(low=np.array([-self.beta/2,-2*self.INERTIA]),
                                               high=np.array([self.beta/2,2*self.INERTIA]),
                                               dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Obstacles
        self.obstacles = []
        if obstacles is None:
            for _ in range(num_obs):
                center = self.observation_space.sample()[:2] # only need x and y, first two indices
                radius = np.random.uniform(obs_low, obs_high)
                self.obstacles.append((center, radius))
        else:
            if not isinstance(obstacles, (tuple, list)):
                raise TypeError("Set of obstacle must be a tuple or list containing tuple or lists of obstacles' centers and radii.")
            if len(obstacles) == 0:
                warnings.warn("No obstacles given.")
            else:
                if not isinstance(obstacles[0], (tuple, list)):
                    raise ValueError("Set of obstacle must be a tuple or list containing tuple or lists of obstacles' centers and radii.")
            for o in obstacles:
                if not isinstance(o, (tuple, list)):
                    raise TypeError("Obstacle must be a tuple or list containing center and radius.")
                if len(o) != 2:
                    raise ValueError("Obstacle must be a tuple or list containing center and radius.")
                if not isinstance(o[0], (tuple, list, np.ndarray)) or len(o[0]) != 2:
                    raise ValueError("Obstacle center must be a tuple, list, or array containing the x, y coordinates of the center.")
                if not isinstance(o[1], (int, float, np.number)):
                    raise ValueError("Obstacle radius must be a single numeric value.")
                self.obstacles.append(o)

        # Goal
        if goal is None:
            center = None
            attempts = 0
            while center is None:
                candidate = self.observation_space.sample()[:2]
                in_obs = any([np.linalg.norm(candidate - c) <= r for c, r in self.obstacles])
                if not in_obs:
                    center = candidate
                attempts += 1
                if attempts == 1e3:
                    warnings.warn("Attempt to find valid goal region is taking a while, you may wish to check input parameters.")
            self.goal = (center, goal_rad)
        else:
            if not isinstance(goal, (tuple, list)):
                raise TypeError("Goal must be a tuple or list containing center and radius.")
            if len(goal) != 2:
                raise ValueError("Goal must be a tuple or list containing center and radius.")
            if not isinstance(goal[0], (tuple, list, np.ndarray)) or len(goal[0]) != 2:
                raise ValueError("Goal center must be a tuple, list, or array containing the x, y coordinates of the center.")
            if not isinstance(goal[1], (int, float, np.number)):
                raise ValueError("Goal radius must be a single numeric value.")
            if any([np.linalg.norm(goal[0] - c) <= r for c, r in self.obstacles]):
                raise ValueError("Goal center may not lie within an obstacle.")
            self.goal = goal
        print(self.goal)

        # Exclusion radius for initial points
        self.goal_exclusion = goal_excl

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
            if "initial_state" in options:
                assert options["initial_state"].shape == self.observation_space.shape
                if any(options["initial_state"] < self.observation_space.low) or any(options["initial_state"] > self.observation_space.high):
                    raise ValueError("Initial state is out of bounds.")
                default_options["initial_state"] = options["initial_state"]
        
        options = default_options

        # Choose the agent's location uniformly at random            
        if "initial_state" in options:
            self._agent_state = options["initial_state"]
        else:
            attempts = 0
            while True:
                # Sample random point in space
                pt = self.np_random.uniform(options["low"],options["high"])
                # Check if it falls in any of the obstacle regions
                in_obs = [np.linalg.norm(pt[:2] - c) <= r for c, r in self.obstacles]
                # Check if it falls in the goal exclusion zone
                in_goal_excl = np.linalg.norm(pt[:2] - self.goal[0]) <= self.goal_exclusion
                # If neither is true, return point
                if not any(in_obs) and not in_goal_excl:
                    self._agent_state = pt
                    break
                attempts += 1
                if attempts == 1e3:
                    warnings.warn("Attempt to find valid initial point is taking a while, you may wish to check input parameters.")
            # self._agent_state = self.np_random.uniform(options["low"],options["high"])
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
        _, _, theta, vx, vy, vtheta = self._agent_state
        # ddx = -self.beta*((u1) + (u2))*np.sin(theta)/self.MASS
        # ddy = self.beta*((u1) + (u2))*np.cos(theta)/self.MASS - self.GRAVITY
        # ddz = self.CENTER_TO_PROP/self.INERTIA*self.beta/2*((u1) - (u2))

        # TODO: check if correct for paper
        control_u1 = np.tanh(self.alpha*u1) if self._options["saturate_controls"] else u1
        control_u2 = np.tanh(self.alpha*u2) if self._options["saturate_controls"] else u2

        noise = np.clip(np.random.normal(0, 1, (3,)), -3, 3) if self.noisy else np.zeros((3,))
        ddx = -self.beta*(self.MASS*self.GRAVITY + control_u1)*np.sin(theta)/self.MASS + noise[0]
        ddy = self.beta*(self.MASS*self.GRAVITY + control_u1)*np.cos(theta)/self.MASS - self.GRAVITY + noise[1]
        ddz = self.beta*control_u2/self.INERTIA + noise[2]        


        euler_step = self.dt*(np.array([vx,vy,vtheta,ddx,ddy,ddz]))

        # Define f = x_k + Delta t * dot(x_k)
        new_state = self._agent_state + euler_step
        new_state[2] = remainder(new_state[2], 2*np.pi) # Keep angle [-pi,pi]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_state = np.clip(
            new_state, self.observation_space.low, self.observation_space.high
        )

        # An episode is done iff the agent has reached the boundary or obstacles
        reach_boundary = np.any(
            (self._agent_state[:2] == self.observation_space.low[:2]) |
            (self._agent_state[:2] == self.observation_space.high[:2])
        )
        reach_obs = any([np.linalg.norm(self._agent_state[:2] - c) <= r for c, r in self.obstacles])
        reach_goal = np.linalg.norm(self._agent_state[:2] - self.goal[0]) <= self.goal[1]
        # truncated is set by a wrapper which I can't access from in here so termination 
        # only occurs for boundary or obstacle hitting, have to check truncation and reward
        # to see if we end in goal state or fail
        terminated = reach_boundary or reach_obs or reach_goal

        # terminated = False
        # reward = 1 if terminated else 0  # Binary sparse rewards
        # reward = 1 if not terminated else -500
        # reward of 1 when in goal region, reward of 0 in free space, reward of -200 in boundary or obstacle        reach_goal = np.linalg.norm(self._agent_state[:2] - self.goal[0]) <= self.goal[1]
        reward = 1 if reach_goal else -200 if (reach_boundary or reach_obs) else 0
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

        # Draw goal region
        x,y = self.goal[0]
        pygame.draw.circle(
            canvas,
            (144, 238, 144),
            pix_per_meter * (np.array([x,-y]) + self.HEIGHT_WIDTH/2),
            self.goal[1]*pix_per_meter
        )

        # Draw obstacles
        for c, r in self.obstacles:
            x,y = c
            pygame.draw.circle(
                canvas,
                (255, 127, 127),
                pix_per_meter * (np.array([x,-y]) + self.HEIGHT_WIDTH/2),
                r*pix_per_meter
            )

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

        # Draw arrows indicating control strength TODO: fix direction to map torque on visual
        arrow_len = self._action[0]*body_width
        clockwise = arrow_len if self._action[1] < 0 else 0
        countercl = arrow_len if self._action[1] > 0 else 0
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
            (body_width - body_height/2, body_height/2 + body_width + self._action[0]*body_width),
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

    def obstacles(self): return self.obstacles

    def goal(self): return self.goal

    def goal_exclusion(self): return self.goal_exclusion

    def in_obstacle(self, pt):
        """
        Returns if the given point lies within an obstacle
        """
        return any([np.linalg.norm(pt[:2] - c) <= r for c, r in self.obstacles])

    def in_goal(self, pt):
        """
        Returns if the given point lies within the goal region
        """
        return np.linalg.norm(pt[:2] - self.goal[0]) <= self.goal[1]