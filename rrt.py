"""Kinodynamic RRT for the planar-quadrotor.
LaValle SM, Kuffner JJ. Randomized Kinodynamic Planning. The
International Journal of Robotics Research. 2001;20(5):378-400.
doi:10.1177/02783640122067453
"""
from __future__ import annotations
from collections import Counter
import pickle
import warnings
import numpy as np
from scipy.optimize import lsq_linear
from planar_quadcopter_env.envs.planar_quadcopter import PlanarQuadCopterEnv
from tqdm import tqdm

N_STEPS = 3     # steps per extension
def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi
def body_axis(theta):
    return np.stack([-np.sin(theta), np.cos(theta)], axis=-1)


# ----------------------------------------------------------------- env dynamics
def env_step(env: PlanarQuadCopterEnv, x: np.ndarray, u: np.ndarray):
    """Set the env to state x, apply u with env.step. Returns (next state, terminated)."""
    env._agent_state = np.array(x, dtype=np.float64)
    obs, _, terminated, _, _ = env.step(u)
    return np.array(obs), bool(terminated)


def rollout(env: PlanarQuadCopterEnv, x: np.ndarray, U: np.ndarray):
    """Apply controls U (k, 2) from x. Returns states after each control (k, 6) and
    the env's `terminated` flag after each step (k,) (True when it hits the box wall)."""
    S, term = np.empty((len(U), 6)), np.zeros(len(U), bool)
    for j, u in enumerate(U):
        x, term[j] = env_step(env, x, u)
        S[j] = x
    return S, term


# ------------------------------------------------------------------ metric rho
def rho_weights(low, high, scale=(1.0, 1.0, 1.0, 1.0)):
    """Paper's normalisation: each term of rho is scaled to [0, 1] over the sampling box.

    Returns (w_p, w_theta, w_v, w_omega). `scale` lets you re-weight the terms.
    """
    span = np.asarray(high, float) - np.asarray(low, float)
    w = np.array([1 / (span[0]**2 + span[1]**2), # largest possible ||dp||^2 in the box
                  1 / np.pi**2, # largest wrapped angle difference
                  1 / (span[3]**2 + span[4]**2),
                  1 / span[5]**2])
    return w * np.asarray(scale, float)


def rho(X: np.ndarray, x: np.ndarray, w) -> float:
    """Weighted distance from each state in X (N, 6) to x (6,).

    rho = w_p ||dp||^2 + w_theta dtheta^2 + w_v ||dv||^2 + w_omega domega^2
    """
    d = X - x
    d[:, 2] = wrap(d[:, 2])
    return (w[0] * np.sum(d[:, :2]**2, -1) + w[1] * d[:, 2]**2
            + w[2] * np.sum(d[:, 3:5]**2, -1) + w[3] * d[:, 5]**2)


# -------------------------------------------------------------------- steering
def steer(env: PlanarQuadCopterEnv, x, x_rand, w):
    """Controls U (3, 2) = [dT, tau] per step from x that bring x_{t+3} as close as
    possible to x_rand in rho, within env.action_space.

    This is a non-convex QP over 6 controls. Use global optimals at each step.
    """
    dt, m, g, inertia = env.dt, env.MASS, env.GRAVITY, env.INERTIA
    lo, hi = env.action_space.low, env.action_space.high
    p, th, v, om = x[:2], x[2], x[3:5], x[5]
    sp, sth, sv, som = np.sqrt(w)

    # 1) torques
    A = np.array([[sth * dt**2 / inertia * 2, sth * dt**2 / inertia, 0.0],
                  [som * dt / inertia] * 3,
                  *(1e-6 * np.eye(3))])
    b = np.array([sth * wrap(x_rand[2] - th - 3 * dt * om), som * (x_rand[5] - om), 0, 0, 0])
    # Theta and omega at t+3 are linear in tau, so we have a convex QP.
    tau = lsq_linear(A, b, bounds=(lo[1], hi[1])).x
    thetas = np.array([th, th + dt * om, th + 2 * dt * om + dt**2 * tau[0] / inertia])

    # 2) thrust deviations
    bk = body_axis(thetas) # (3, 2)
    a0 = g * bk - g * np.array([0.0, 1.0]) # acceleration with dT = 0
    c = bk / m # d a_k / d dT_k
    A = np.vstack([sp * dt**2 * np.stack([2 * c[0], c[1], np.zeros(2)], axis=1),
                   sv * dt * np.stack([c[0], c[1], c[2]], axis=1)])
    b = np.concatenate([sp * (x_rand[:2] - p - 3 * dt * v - dt**2 * (2 * a0[0] + a0[1])),
                        sv * (x_rand[3:5] - v - dt * a0.sum(axis=0))])
    # tau is fixed, so we have a convex QP for position and velocity at t+3
    dT = lsq_linear(A, b, bounds=(lo[0], hi[0])).x
    return np.stack([dT, tau], axis=1)


# ----------------------------------------------------------------------- tree
class Node:
    def __init__(self, state: np.ndarray):
        self.state = state
        self.parent: Node | None = None
        self.to_nodes: dict[Node, np.ndarray] = {}
        self.control = np.zeros(2) # control [dT, tau] applied at parent to reach this node
        self.depth = 0 # number of steps from the root

    def add_parent(self, node: Node):
        self.parent = node
        self.depth = node.depth + 1

    def add_to_node(self, node: Node, controls: np.ndarray):
        self.to_nodes[node] = controls


class RRT:
    # RANDOM_STATE box for [theta, v_x, v_y, omega]; (x, y) comes from env.observation_space.
    SAMPLE_LOW_REST = np.array([-np.pi / 4, -2.0, -2.0, -1.0])
    SAMPLE_HIGH_REST = np.array([np.pi / 4, 2.0, 2.0, 1.0])
    RHO_SCALE = (10.0, 10.0, 1.0, 1.0)     # extra multipliers on (w_p, w_theta, w_v, w_omega)

    EXPLORE_MODES = ("all", "last")

    def __init__(self, env, x_init, is_free, is_goal, max_iter=75000, options=None, seed=None,
                 explore_from="all"):
        """
        env: planar_quadcopter env, is modified to be not noisy.
        is_free:  state -> True if not in an obstacle and not in the goal.
        is_goal:  state -> True if in the goal.
        A state is allowed if is_free or is_goal.
        explore_from: which states of each N_STEPS extension NEAREST_NEIGHBOR may extend from.
            "all":  every intermediate state (x_{t+1}, x_{t+2}, x_{t+3}).
            "last": only the final state x_{t+3}.
            Either way all intermediate states stay in the tree (self.nodes, parents),
            so goal paths still have one state and one control per env step.
        """
        if explore_from not in self.EXPLORE_MODES:
            raise ValueError(f"explore_from must be one of {self.EXPLORE_MODES}, got {explore_from!r}")
        self.explore_from = explore_from
        self.env: PlanarQuadCopterEnv = env.unwrapped
        # assert self.env.beta == 1, "steer assumes the paper's dynamics (env.beta == 1)"
        x_init = np.asarray(x_init, float)
        # self.env.noisy = False
        self.env.reset(options={"low": x_init, "high": x_init, "saturate_controls": False, "initial_state": x_init})

        self.is_free, self.is_goal, self.max_iter = is_free, is_goal, max_iter
        assert self.allowed(x_init), f"x_init {x_init} is in an obstacle"
        self.rng = np.random.default_rng(seed)
        obs = self.env.observation_space
        # self.sample_low = np.concatenate([obs.low[:2], self.SAMPLE_LOW_REST])
        # self.sample_high = np.concatenate([obs.high[:2], self.SAMPLE_HIGH_REST])
        self.sample_low = obs.low
        self.sample_high = obs.high
        self.w = rho_weights(self.sample_low, self.sample_high, self.RHO_SCALE)

        self.head = Node(x_init)
        self.nodes: list[Node] = []
        self._X = np.empty((N_STEPS * max_iter + 1, 6))   # node states for NEAREST_NEIGHBOR
        self._extendable = np.zeros(N_STEPS * max_iter + 1, bool) # states not in goal region
        self.add_node(self.head)

        self.goal_nodes: list[Node] = []

        # Diagnostic variables, in case you want them.
        self.samples: list[np.ndarray] = []
        self.n_trapped = 0

    # ------------------------------------------------------------- pickling
    # Lets an RRT be returned from a ProcessPoolExecutor worker. The tree is stored flat
    # (parent indices) because pickling the linked Nodes recurses once per tree level and
    # overflows on deep trees. is_free / is_goal that cannot be pickled (e.g. lambdas or
    # closures defined inside a function) are dropped; the unpickled tree, paths and
    # nodes are intact, but plan()/extend() need them set again (rrt.is_free = ...).
    def __getstate__(self):
        state = self.__dict__.copy()
        nodes = state.pop("nodes")
        del state["head"], state["goal_nodes"]
        n = len(nodes)
        index = {id(node): i for i, node in enumerate(nodes)}
        state["_tree"] = {
            "parent": np.array([-1 if nd.parent is None else index[id(nd.parent)] for nd in nodes], dtype=np.int64),
            "control": np.array([nd.control for nd in nodes]).reshape(n, 2),
            "goal": np.array([index[id(g)] for g in self.goal_nodes], dtype=np.int64),
            "capacity": len(self._X),
        }
        state["_X"], state["_extendable"] = self._X[:n].copy(), self._extendable[:n].copy()
        for name in ("is_free", "is_goal"):
            try:
                pickle.dumps(state[name])
            except Exception:
                warnings.warn(f"RRT.{name} is not picklable and was dropped; set it again "
                              "before calling plan()/extend() on the unpickled RRT.")
                state[name] = None
        return state

    def __setstate__(self, state):
        tree = state.pop("_tree")
        self.__dict__.update(state)
        n, cap = len(self._X), tree["capacity"]
        X, ext = np.empty((cap, 6)), np.zeros(cap, bool)
        X[:n], ext[:n] = self._X, self._extendable
        self._X, self._extendable = X, ext
        self.nodes = []
        for i in range(n):  # parents always precede their children in self.nodes
            node = Node(X[i].copy())
            node.control = tree["control"][i].copy()
            p = tree["parent"][i]
            if p >= 0:
                node.add_parent(self.nodes[p])
                self.nodes[p].add_to_node(node, node.control)
            self.nodes.append(node)
        self.head = self.nodes[0]
        self.goal_nodes = [self.nodes[i] for i in tree["goal"]]

    def allowed(self, x) -> bool:
        return self.is_free(x) or self.is_goal(x)

    def add_node(self, node: Node, explorable: bool = True):
        """Add a node to the graph. explorable=False keeps it in the tree (for paths)
        but hides it from NEAREST_NEIGHBOR."""
        n = len(self.nodes)
        self._X[n] = node.state
        # Mark nodes not to continue from.
        self._extendable[n] = explorable and not self.is_goal(node.state)
        self.nodes.append(node)
        if node.parent is not None:
            node.parent.add_to_node(node, node.control)

    def random_state(self) -> np.ndarray:
        while True:     # samples in obstacles are redrawn and not counted
            x = self.rng.uniform(self.sample_low, self.sample_high)
            if self.allowed(x):
                return x

    def nearest_neighbor(self, x: np.ndarray) -> Node:
        n = len(self.nodes)
        d = rho(self._X[:n], x, self.w)
        # Neighbors that are are in the goal region are ignored.
        d[~self._extendable[:n]] = np.inf
        return self.nodes[int(np.argmin(d))]

    def new_state(self, x_rand: np.ndarray, x_near: Node):
        """Controls (k, 2) and states (k, 6) pulling x_near toward x_rand, cut at the
        first goal state (k <= 3); None if trapped."""
        U = steer(self.env, x_near.state, x_rand, self.w)
        S, terminated = rollout(self.env, x_near.state, U)
        for j, s in enumerate(S):
            # The env also sets terminated in the goal, so check the goal first.
            if self.is_goal(s):
                return U[: j + 1], S[: j + 1]
            if terminated[j] or not self.is_free(s):
                return None
        return U, S

    def extend(self, x_rand: np.ndarray) -> list[Node]:
        """EXTEND(T, x_rand). Returns the added nodes in time order; empty means Trapped."""
        x_near = self.nearest_neighbor(x_rand)
        result = self.new_state(x_rand, x_near)
        if result is None:
            self.n_trapped += 1
            return []
        new_nodes, parent = [], x_near
        U, S = result
        for j, (u, s) in enumerate(zip(U, S)):
            child = Node(s.copy())
            child.control = u.copy()
            child.add_parent(parent)
            self.add_node(child, explorable=self.explore_from == "all" or j == len(S) - 1)
            new_nodes.append(child)
            parent = child
        if self.is_goal(parent.state):
            self.goal_nodes.append(parent)
        return new_nodes

    def plan(self, min_paths=np.inf, position=0, disable_tqdm=False):
        """BUILD_RRT for all max_iter iterations. Returns the list of goal paths,
        each (states (T+1, 6), controls (T, 2))."""
        if self.is_goal(self.head.state):
            self.goal_nodes.append(self.head)
            return self.goal_paths()
        for _ in tqdm(range(self.max_iter), desc="RRT iterations", unit="iter", ncols=80, leave=False,position=position,disable=disable_tqdm):
            # Early stopping.
            if len(self.goal_paths()) >= min_paths:
                tqdm.write(f"Stopped early found {len(self.goal_paths())} paths")
                break
            x_rand = self.random_state()
            self.samples.append(x_rand)
            self.extend(x_rand)
        return self.goal_paths()

    def extract_path(self, node: Node):
        # From node, get path going through parents and return reverse order.
        states, controls = [node.state], []
        while node.parent is not None:
            controls.append(node.control)
            node = node.parent
            states.append(node.state)
        return np.array(states[::-1]), np.array(controls[::-1])

    def goal_paths(self):
        return [self.extract_path(n) for n in self.goal_nodes]

    def most_common_length_paths(self):
        """(T, paths) for the path length T (in steps) that occurs most often among
        the goal paths; ties go to the shorter T. Returns (None, []) if no goal paths."""
        if not self.goal_nodes:
            return None, []
        counts = Counter(n.depth for n in self.goal_nodes)
        T = min(counts, key=lambda t: (-counts[t], t))
        return T, [self.extract_path(n) for n in self.goal_nodes if n.depth == T]
