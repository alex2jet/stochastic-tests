import jax
import jax.numpy as jnp
import optax

from jax.scipy.linalg import expm
from jax import random, grad, jit, vmap

# ============================================================
# Constants
# ============================================================

kB = 1.380649e-23

# ============================================================
# State-space construction
# ============================================================

def build_A(omega0, gamma):
    return jnp.array([
        [0.0, 1.0],
        [-omega0**2, -gamma]
    ])


def build_B(gamma, temperature, mass=1.0):
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
# Kalman filter
# ============================================================

def kalman_filter(y, F, Q, R):

    H = jnp.array([[1.0, 0.0]])

    T = y.shape[0]

    # initial state
    m = jnp.zeros((2,))
    P = jnp.eye(2)

    loglik = 0.0

    filtered_means = []
    filtered_covs = []

    for t in range(T):

        # Prediction
        m_pred = F @ m
        P_pred = F @ P @ F.T + Q

        # Innovation
        S = H @ P_pred @ H.T + R

        K = P_pred @ H.T @ jnp.linalg.inv(S)

        residual = y[t] - H @ m_pred

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

        filtered_means.append(m)
        filtered_covs.append(P)

    return (
        loglik,
        jnp.stack(filtered_means),
        jnp.stack(filtered_covs)
    )


# ============================================================
# Variational parameters
# ============================================================

def init_variational_params():

    return {
        "mu_log_omega0": jnp.array(0.0),
        "rho_log_omega0": jnp.array(-3.0),

        "mu_log_gamma": jnp.array(-1.0),
        "rho_log_gamma": jnp.array(-3.0),

        "mu_log_T": jnp.array(1.0),
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

# observed data
# shape: (T,)
y = jnp.load("trajectory.npy")

params = init_variational_params()

optimizer = optax.adam(1e-2)

opt_state = optimizer.init(params)

key = random.PRNGKey(0)

for step in range(5000):

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

        gamma_est = jnp.exp(
            params["mu_log_gamma"]
        )

        T_est = jnp.exp(
            params["mu_log_T"]
        )

        print(
            step,
            loss,
            omega_est,
            gamma_est,
            T_est
        )