import jax
import jax.numpy as jnp
import optax

from jax.scipy.linalg import expm
from jax import random, grad, jit

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Constants
# ============================================================

kB = 1.380649e-23
# A small oscillator mass keeps the thermal displacement on the scale of the
# observation noise used in this example.
MASS = 1e-20

# ============================================================
# State-space construction
# ============================================================

def build_A(omega0, gamma):
    return jnp.array([
        [0.0, 1.0],
        [-omega0**2, -gamma]
    ])


def build_B(gamma, temperature, mass=MASS):
    sigma = jnp.sqrt(2.0 * gamma * kB * temperature / mass)

    return jnp.array([
        [0.0],
        [sigma]
    ])


# ============================================================
# Discretization
# ============================================================

def discretize_system(omega0, gamma, temperature, dt):

    A = build_A(omega0, gamma)
    B = build_B(gamma, temperature)

    # State transition matrix
    F = expm(A * dt)

    # Approximate Q using midpoint rule
    # Better: Van Loan method
    M = expm(A * dt / 2.0)
    Q = dt * (M @ B @ B.T @ M.T)

    return F, Q


# ============================================================
# Synthetic observations
# ============================================================

def generate_synthetic_data(
    key,
    num_steps,
    dt,
    R,
    omega0=2 * np.pi * 2.0,
    gamma=0.4,
    temperature=300.0,
):
    """Simulate noisy position observations from the oscillator model."""
    F, _ = discretize_system(omega0, gamma, temperature, dt)

    # The midpoint process covariance has the form g @ g.T, so sampling one
    # scalar normal per step produces process noise with exactly that covariance.
    A = build_A(omega0, gamma)
    B = build_B(gamma, temperature)
    M = expm(A * dt / 2.0)
    process_noise_scale = (jnp.sqrt(dt) * M @ B).reshape(2,)

    process_key, observation_key = random.split(key)
    process_noise = random.normal(process_key, (num_steps,))
    observation_noise = random.normal(observation_key, (num_steps,)) * jnp.sqrt(R[0, 0])

    def step(state, noise):
        state = F @ state + process_noise_scale * noise
        return state, state[0]

    _, positions = jax.lax.scan(step, jnp.zeros((2,)), process_noise)
    return positions + observation_noise


# ============================================================
# Kalman filter
# ============================================================

def kalman_filter(y, F, Q, R):

    H = jnp.array([[1.0, 0.0]])

    # initial state
    initial_state = (jnp.zeros((2,)), jnp.eye(2), 0.0)

    def filter_step(carry, observation):
        m, P, loglik = carry
        # Prediction
        m_pred = F @ m
        P_pred = F @ P @ F.T + Q

        # Innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ jnp.linalg.inv(S)
        residual = observation - H @ m_pred

        # Update
        m = m_pred + (K @ residual).reshape(-1)
        P = P_pred - K @ H @ P_pred

        # Log likelihood
        logdetS = jnp.log(S[0, 0])
        quad = residual.T @ jnp.linalg.inv(S) @ residual

        loglik += -0.5 * (
            logdetS +
            quad +
            jnp.log(2.0 * jnp.pi)
        )

        return (m, P, loglik), (m, P)

    (_, _, loglik), (filtered_means, filtered_covs) = jax.lax.scan(
        filter_step,
        initial_state,
        y,
    )
    return (
        loglik,
        filtered_means,
        filtered_covs,
    )


# ============================================================
# Variational parameters
# ============================================================

def init_variational_params():
    return {
        "mu_log_omega0": jnp.array(jnp.log(2 * np.pi)),
        "rho_log_omega0": jnp.array(-3.0),

        "mu_log_gamma": jnp.array(jnp.log(0.2)),
        "rho_log_gamma": jnp.array(-3.0),

        "mu_log_T": jnp.array(jnp.log(300.0)),
        "rho_log_T": jnp.array(-3.0),
    }


# ============================================================
# Softplus transform
# ============================================================

def softplus(x):
    return jnp.log1p(jnp.exp(x))


# ============================================================
# Reparameterized sampling
# ============================================================

def sample_params(params, key):

    k1, k2, k3 = random.split(key, 3)

    eps1 = random.normal(k1)
    eps2 = random.normal(k2)
    eps3 = random.normal(k3)

    s1 = softplus(params["rho_log_omega0"])
    s2 = softplus(params["rho_log_gamma"])
    s3 = softplus(params["rho_log_T"])

    log_omega0 = (
        params["mu_log_omega0"] +
        s1 * eps1
    )

    log_gamma = (
        params["mu_log_gamma"] +
        s2 * eps2
    )

    log_T = (
        params["mu_log_T"] +
        s3 * eps3
    )

    omega0 = jnp.exp(log_omega0)
    gamma = jnp.exp(log_gamma)
    temperature = jnp.exp(log_T)

    return omega0, gamma, temperature


# ============================================================
# KL divergence to standard normal prior
# ============================================================

def gaussian_kl(mu, rho):
    sigma = softplus(rho)

    return 0.5 * (
        mu**2 +
        sigma**2 -
        2.0 * jnp.log(sigma) -
        1.0
    )


# ============================================================
# ELBO
# ============================================================

def elbo(params, y, dt, R, key):
    omega0, gamma, temperature = sample_params(params, key)

    F, Q = discretize_system(
        omega0,
        gamma,
        temperature,
        dt
    )

    loglik, _, _ = kalman_filter(y, F, Q, R)

    kl = (
        gaussian_kl(
            params["mu_log_omega0"],
            params["rho_log_omega0"]
        )
        +
        gaussian_kl(
            params["mu_log_gamma"],
            params["rho_log_gamma"]
        )
        +
        gaussian_kl(
            params["mu_log_T"],
            params["rho_log_T"]
        )
    )

    return loglik - kl


# ============================================================
# Optimization step
# ============================================================

@jit
def train_step(params, opt_state, y, dt, R, key):
    loss_fn = lambda p: -elbo(p, y, dt, R, key)
    grads = grad(loss_fn)(params)

    updates, opt_state = optimizer.update(
        grads,
        opt_state
    )

    params = optax.apply_updates(params, updates)
    loss = loss_fn(params)

    return params, opt_state, loss


# ============================================================
# Example usage
# ============================================================

dt = 0.01
R = jnp.array([[1e-4]])

# Generate observations in memory (shape: (T,)); no intermediate file is used.
key = random.PRNGKey(0)
data_key, key = random.split(key)
y = generate_synthetic_data(
    data_key,
    num_steps=1000,
    dt=dt,
    R=R,
)

time = np.arange(y.shape[0]) * dt

plt.figure(figsize=(10, 4))
plt.plot(time, np.asarray(y), linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.title("Synthetic oscillator observations")
plt.grid(True)
plt.tight_layout()
plt.show()

# exit()

params = init_variational_params()
schedule = optax.exponential_decay(
    init_value=1e-2,
    transition_steps=200,
    decay_rate=0.7,
    staircase=True,
    end_value=1e-4,
)
optimizer = optax.adam(learning_rate=schedule)
opt_state = optimizer.init(params)

for step in range(2000):
    key, subkey = random.split(key)
    params, opt_state, loss = train_step(
        params,
        opt_state,
        y,
        dt,
        R,
        subkey
    )

    if step % 100 == 0:
        omega_est = jnp.exp(
            params["mu_log_omega0"]
        )
        omega_rho = jnp.exp(
            params["rho_log_omega0"]
        )

        gamma_est = jnp.exp(
            params["mu_log_gamma"]
        )
        gamma_rho = jnp.exp(
            params["rho_log_gamma"]
        )

        T_est = jnp.exp(
            params["mu_log_T"]
        )
        T_rho = jnp.exp(
            params["rho_log_T"]
        )

        print(
            step,
            loss,
            omega_est,
            omega_rho,
            gamma_est,
            gamma_rho,
            T_est,
            T_rho,
        )
