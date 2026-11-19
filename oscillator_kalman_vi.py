"""Collapsed Kalman + variational inference for one thermal oscillator.

Install: python -m pip install jax numpyro
Run the synthetic example: python oscillator_kalman_vi.py

For measured data, enable JAX float64 before constructing the data object:

    from math import tau

    import jax
    from oscillator_kalman_vi import OscillatorData, fit_oscillator, smooth_posterior

    jax.config.update("jax_enable_x64", True)
    data = OscillatorData(y, dt=dt, mass=mass, gamma=gamma, sigma_y=sigma_y)
    fit = fit_oscillator(jax.random.key(0), data,
                         prior_omega=tau * f_guess,
                         prior_temperature=T_guess)
    draws = fit.sample(jax.random.key(1), num_samples=256)
    trajectory = smooth_posterior(data, draws)

Here y is a 1-D sequence of equally spaced position measurements in meters,
relative to a known equilibrium position. dt is in seconds, mass in kg,
gamma in 1/s, and sigma_y is the known measurement standard deviation in
meters. The damping convention is x'' + gamma*x' + omega**2*x = F/m, so
the deterministic underdamped amplitude envelope decays as exp(-gamma*t/2).
omega is the undamped natural angular frequency in rad/s and T is in kelvin.

The initial state is at thermal equilibrium. For the state [x, v], the
continuous-time driving term is sqrt(2*gamma*k_B*T/m) dW in velocity.
NumPyro fits one two-dimensional Gaussian for [log(omega), log(T)], using
the exact innovation log likelihood and the analytic Gaussian KL. The RTS
smoother is used after fitting to estimate position and velocity.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, TraceMeanField_ELBO
from numpyro.optim import Adam


BOLTZMANN = 1.380649e-23  # J/K


@dataclass(frozen=True)
class OscillatorData:
    y: jax.Array
    dt: float
    mass: float
    gamma: float
    sigma_y: float

    def __post_init__(self):
        if not jax.config.x64_enabled:
            raise RuntimeError('Enable double precision with jax.config.update("jax_enable_x64", True).')
        y = np.asarray(self.y, dtype=np.float64)
        if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
            raise ValueError("y must be a nonempty, finite 1-D array of positions.")
        for name in ("dt", "mass", "gamma", "sigma_y"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "y", jnp.asarray(y))


class StateMarginals(NamedTuple):
    # Shapes are (n_time, 2) and (n_time, 2, 2). State order is [x, v] in SI.
    mean: jax.Array
    covariance: jax.Array


class VariationalFit(NamedTuple):
    log_location: jax.Array
    log_scale_tril: jax.Array
    losses: jax.Array  # Negative Monte Carlo ELBO per optimizer step.

    def sample(self, key, num_samples=1000):
        """Return an array of shape (num_samples, 2), ordered [omega, T]."""
        posterior = dist.MultivariateNormal(
            self.log_location, scale_tril=self.log_scale_tril
        )
        return jnp.exp(posterior.sample(key, (num_samples,)))


def _discretize(omega, temperature, dt, mass, gamma, sigma_y):
    # Internally z = [x/sigma_y, v/(omega*sigma_y)]. The observation noise
    # has unit variance and thermal equilibrium has covariance variance*I.
    variance = BOLTZMANN * temperature / (mass * (omega * sigma_y) ** 2)
    drift = jnp.array([[0.0, omega], [-omega, -gamma]])
    transition = jsl.expm(dt * drift)

    # For a transition started at a known state, evolve [Q00,Q01,Q11]/variance
    # from zero using Q' = A Q + Q A.T + diag(0, 2*gamma)*variance.
    # The final coordinate in this augmented ODE is a constant 1.
    # This avoids subtracting nearly equal stationary covariances for small dt.
    covariance_ode = jnp.array([
        [0.0, 2.0 * omega, 0.0, 0.0],
        [-omega, -gamma, omega, 0.0],
        [0.0, -2.0 * omega, -2.0 * gamma, 2.0 * gamma],
        [0.0, 0.0, 0.0, 0.0],
    ])
    q00, q01, q11 = jsl.expm(dt * covariance_ode)[:3, 3]
    process_cov = variance * jnp.array([[q00, q01], [q01, q11]])
    return transition, process_cov, variance * jnp.eye(2)


def _symmetrize(cov):
    return (cov + cov.T) * 0.5


def _filter(omega, temperature, y, dt, mass, gamma, sigma_y, *, save_history):
    transition, process_cov, initial_cov = _discretize(
        omega, temperature, dt, mass, gamma, sigma_y
    )
    identity = jnp.eye(2)
    log_normalizer = jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(sigma_y)

    def step(carry, observation):
        predicted_mean, predicted_cov, log_likelihood = carry
        innovation = observation - predicted_mean[0]
        innovation_var = predicted_cov[0, 0] + 1.0
        gain = predicted_cov[:, 0] / innovation_var
        filtered_mean = predicted_mean + gain * innovation

        # Joseph update with normalized observation row [1, 0] and R=1.
        correction = identity.at[:, 0].add(-gain)
        filtered_cov = _symmetrize(
            correction @ predicted_cov @ correction.T + jnp.outer(gain, gain)
        )
        log_likelihood -= 0.5 * (
            log_normalizer + jnp.log(innovation_var) + innovation**2 / innovation_var
        )
        next_mean = transition @ filtered_mean
        next_cov = _symmetrize(transition @ filtered_cov @ transition.T + process_cov)
        history = (predicted_mean, predicted_cov, filtered_mean, filtered_cov)
        return (next_mean, next_cov, log_likelihood), history if save_history else None

    initial = (jnp.zeros(2), initial_cov, jnp.zeros(()))
    (_, _, log_likelihood), history = jax.lax.scan(step, initial, y / sigma_y)
    return log_likelihood, transition, history


@jax.jit
def _log_likelihood(omega, temperature, y, dt, mass, gamma, sigma_y):
    return _filter(
        omega, temperature, y, dt, mass, gamma, sigma_y, save_history=False
    )[0]


def _data_args(data):
    return data.y, data.dt, data.mass, data.gamma, data.sigma_y


def log_likelihood(data, omega, temperature):
    """Return log p(y | omega, T) for positive scalar omega and T."""
    return _log_likelihood(omega, temperature, *_data_args(data))


def _model(y, dt, mass, gamma, sigma_y, prior_location, prior_scale):
    log_theta = numpyro.sample(
        "log_theta",
        dist.MultivariateNormal(prior_location, scale_tril=jnp.diag(prior_scale)),
    )
    omega, temperature = jnp.exp(log_theta)
    numpyro.factor(
        "position_log_likelihood",
        _log_likelihood(omega, temperature, y, dt, mass, gamma, sigma_y),
    )


def _guide(y, dt, mass, gamma, sigma_y, prior_location, prior_scale):
    location = numpyro.param("log_location", prior_location)
    scale_tril = numpyro.param(
        "log_scale_tril", jnp.diag(0.25 * prior_scale),
        constraint=constraints.lower_cholesky,
    )
    numpyro.sample("log_theta", dist.MultivariateNormal(location, scale_tril=scale_tril))


def fit_oscillator(
    key, data, *, prior_omega, prior_temperature,
    log_prior_scale=(0.35, 0.7), num_steps=1500, num_particles=8,
    learning_rate=0.02, progress_bar=True,
):
    """Fit q(log(omega), log(T)) by stochastic variational inference.

    prior_omega and prior_temperature are prior medians in rad/s and kelvin.
    log_prior_scale gives the two prior standard deviations in natural log units.
    num_particles is the number of parameter draws per gradient step.
    """
    medians = np.asarray([prior_omega, prior_temperature], dtype=np.float64)
    scales = np.asarray(log_prior_scale, dtype=np.float64)
    if (medians.shape != (2,) or scales.shape != (2,) or
            not np.all(np.isfinite(medians)) or not np.all(np.isfinite(scales)) or
            np.any(medians <= 0) or np.any(scales <= 0)):
        raise ValueError("The prior medians and both log scales must be finite and positive.")
    if not all(isinstance(n, (int, np.integer)) and n > 0
               for n in (num_steps, num_particles)):
        raise ValueError("num_steps and num_particles must be positive integers.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and strictly positive.")

    # Both model and guide have one latent site, so even a correlated Gaussian
    # for that site satisfies the ELBO estimator's factorization requirement.
    svi = SVI(_model, _guide, Adam(learning_rate),
              TraceMeanField_ELBO(num_particles=num_particles))
    result = svi.run(
        key, num_steps, *_data_args(data), jnp.log(medians), jnp.asarray(scales),
        progress_bar=progress_bar,
    )
    if not np.all(np.isfinite(np.asarray(result.losses))):
        raise FloatingPointError("Nonfinite ELBO. Check units, prior range, and learning rate.")
    return VariationalFit(
        result.params["log_location"], result.params["log_scale_tril"], result.losses
    )


def _smooth(omega, temperature, y, dt, mass, gamma, sigma_y):
    _, transition, history = _filter(
        omega, temperature, y, dt, mass, gamma, sigma_y, save_history=True
    )
    predicted_mean, predicted_cov, filtered_mean, filtered_cov = history

    def backward(carry, inputs):
        next_mean, next_cov = carry
        mean, cov, next_predicted_mean, next_predicted_cov = inputs
        gain = jsl.solve(next_predicted_cov, transition @ cov, assume_a="pos").T
        smoothed_mean = mean + gain @ (next_mean - next_predicted_mean)
        smoothed_cov = _symmetrize(
            cov + gain @ (next_cov - next_predicted_cov) @ gain.T
        )
        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    final = (filtered_mean[-1], filtered_cov[-1])
    if y.size == 1:
        means, covariances = filtered_mean, filtered_cov
    else:
        _, (means, covariances) = jax.lax.scan(
            backward, final, (filtered_mean[:-1], filtered_cov[:-1],
                              predicted_mean[1:], predicted_cov[1:]), reverse=True,
        )
        means = jnp.concatenate((means, filtered_mean[-1:]), axis=0)
        covariances = jnp.concatenate((covariances, filtered_cov[-1:]), axis=0)

    scale = sigma_y * jnp.array([1.0, omega])
    return StateMarginals(means * scale, covariances * jnp.outer(scale, scale))


_smooth_jit = jax.jit(_smooth)


def smooth_conditional(data, omega, temperature):
    """Return exact smoothed marginals conditional on positive scalar omega and T."""
    return _smooth_jit(omega, temperature, *_data_args(data))


@jax.jit
def _smooth_draws(draws, y, dt, mass, gamma, sigma_y):
    # Online mixture moments use memory proportional to the trajectory length.
    def accumulate(carry, theta):
        count, mean, covariance_sum = carry
        state = _smooth(*theta, y, dt, mass, gamma, sigma_y)
        count = count + 1
        delta = state.mean - mean
        mean = mean + delta / count
        between = jnp.einsum("ti,tj->tij", delta, delta) * ((count - 1) / count)
        covariance_sum = covariance_sum + state.covariance + between
        return (count, mean, covariance_sum), None

    n = y.size
    initial = (jnp.zeros(()), jnp.zeros((n, 2)), jnp.zeros((n, 2, 2)))
    (count, mean, covariance_sum), _ = jax.lax.scan(accumulate, initial, draws)
    return StateMarginals(mean, covariance_sum / count)


def smooth_posterior(data, draws):
    """Average the RTS marginals over parameter draws of shape (n_draws, 2).

    Returns the mean and the total covariance at each observation time.
    These moments describe a Gaussian mixture and include parameter uncertainty.
    Cross-time covariances are outside the returned per-time marginals.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if (draws.ndim != 2 or draws.shape[1] != 2 or draws.shape[0] == 0 or
            not np.all(np.isfinite(draws)) or np.any(draws <= 0)):
        raise ValueError("draws must be a nonempty array of positive [omega, T] rows.")
    return _smooth_draws(jnp.asarray(draws), *_data_args(data))


def _simulate(key, n, omega, temperature, dt, mass, gamma, sigma_y):
    """Simulate the same exact sampled model used by the likelihood."""
    initial_key, state_key, observation_key = jax.random.split(key, 3)
    transition, process_cov, initial_cov = _discretize(
        omega, temperature, dt, mass, gamma, sigma_y
    )
    initial = jnp.linalg.cholesky(initial_cov) @ jax.random.normal(initial_key, (2,))
    process_cholesky = jnp.linalg.cholesky(process_cov)

    def propagate(state, white_noise):
        next_state = transition @ state + process_cholesky @ white_noise
        return next_state, next_state

    _, states = jax.lax.scan(
        propagate, initial, jax.random.normal(state_key, (n - 1, 2))
    )
    states = jnp.concatenate((initial[None, :], states), axis=0)
    y = sigma_y * (states[:, 0] + jax.random.normal(observation_key, (n,)))
    return y, states * (sigma_y * jnp.array([1.0, omega]))


def main():
    jax.config.update("jax_enable_x64", True)
    simulate_key, fit_key, sample_key = jax.random.split(jax.random.key(2026), 3)
    dt, mass, gamma, sigma_y = 0.025, 1e-15, 1.2, 7e-6
    true_omega, true_temperature = 2.0 * np.pi * 3.0, 300.0
    y, states = _simulate(
        simulate_key, 1200, true_omega, true_temperature, dt, mass, gamma, sigma_y
    )
    data = OscillatorData(y, dt, mass, gamma, sigma_y)
    fit = fit_oscillator(
        fit_key, data, prior_omega=2.0 * np.pi * 2.8, prior_temperature=280.0
    )
    draws = fit.sample(sample_key, num_samples=256)
    marginal = smooth_posterior(data, draws)
    intervals = np.quantile(np.asarray(draws), [0.025, 0.5, 0.975], axis=0)
    for label, truth, (lower, median, upper) in zip(
        ("omega (rad/s)", "temperature (K)"),
        (true_omega, true_temperature), intervals.T,
    ):
        print(f"{label}: true {truth:.3f}, VI median {median:.3f}, "
              f"95% VI interval [{lower:.3f}, {upper:.3f}]")

    raw_rmse = jnp.sqrt(jnp.mean((data.y - states[:, 0]) ** 2))
    smoothed_rmse = jnp.sqrt(jnp.mean((marginal.mean[:, 0] - states[:, 0]) ** 2))
    print(f"Position RMSE: measured {raw_rmse:.3e} m, smoothed {smoothed_rmse:.3e} m")


if __name__ == "__main__":
    main()
