import star.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from star.pinn import PINN
from star.network import Network
from star.optimizer import L_BFGS_B

# --- Traveling sine wave packet ---
def u0(tx):
    """
    Initial wave form: sine wave packet.
    """
    x = tx[..., 1, None]
    # Sine wave packet centered at x0 with width sigma
    x0 = 0.5
    sigma = 0.1
    k = 5*np.pi   # wave number
    return tf.sin(k*(x-x0)) * tf.exp(-((x-x0)**2)/(2*sigma**2))

def du0_dt(tx):
    """
    Initial velocity (time derivative) of the wave.
    """
    # traveling to the right: du/dt = -c * du/dx
    c = 1.0
    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dx = g.batch_jacobian(u, tx)[..., 1]
    return -c * du_dx

if __name__ == '__main__':
    num_train_samples = 10000
    num_test_samples = 1000

    # build network and PINN
    network = Network.build()
    network.summary()
    pinn = PINN(network).build()

    # --- Training data ---
    # PDE collocation points
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = tx_eqn[..., 0] * 4.0      # t in [0,4]
    tx_eqn[..., 1] = tx_eqn[..., 1]             # x in [0,1]

    # Initial condition points
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                          # t=0
    tx_ini[..., 1] = tx_ini[..., 1]             # x in [0,1]

    # Boundary condition points (free ends: du/dx=0)
    tx_bnd = np.random.rand(num_train_samples, 2)
    tx_bnd[..., 0] = tx_bnd[..., 0] * 4.0       # t in [0,4]
    tx_bnd[..., 1] = np.round(tx_bnd[..., 1])   # x=0 or 1

    # Training outputs
    u_zero = np.zeros((num_train_samples, 1))       # PDE residual
    u_ini = u0(tf.constant(tx_ini)).numpy()        # initial displacement
    du_dt_ini = du0_dt(tf.constant(tx_ini)).numpy() # initial velocity
    du_dx_bnd = np.zeros((num_train_samples, 1))    # free ends: du/dx=0

    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_zero, u_ini, du_dt_ini, du_dx_bnd]

    # Train using L-BFGS-B
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    start_time = time.perf_counter()
    lbfgs.fit()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"\nTraining time: {minutes} min {seconds:.2f} sec")

    # --- Prediction ---
    t_flat = np.linspace(0, 4, num_test_samples)
    x_flat = np.linspace(0, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u_pred = network.predict(tx, batch_size=num_test_samples)
    u_pred = u_pred.reshape(t.shape)

    # --- True solution for a traveling wave (if linear) ---
    # For comparison: right-going wave, exact if linear
    x0 = 0.5
    sigma = 0.1
    k = 5*np.pi
    c = 1.0
    u_true = np.sin(k*(x - c*t - x0)) * np.exp(-((x - c*t - x0)**2)/(2*sigma**2))

    # --- L2 error ---
    l2_error = np.sqrt(np.mean((u_pred - u_true) ** 2))
    print(f"L2 error: {l2_error:.6e}")

    # --- Visualization ---
    fig = plt.figure(figsize=(8,4))
    gs = GridSpec(2, 4)

    # full t-x distribution
    plt.subplot(gs[0, :])
    vmin, vmax = -1, 1
    plt.pcolormesh(t, x, u_pred, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(vmin, vmax)

    # cross-sections
    t_cross_sections = [1, 2, 3, 4]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx_cs = np.stack([np.full_like(x_flat, t_cs), x_flat], axis=-1)
        u_cs = network.predict(tx_cs, batch_size=num_test_samples)
        u_true_cs = np.sin(k * (x_flat - c * t_cs - x0)) * np.exp(-((x_flat - c * t_cs - x0) ** 2) / (2 * sigma ** 2))

        plt.plot(x_flat, u_cs, 'r-', label='PINN')
        plt.plot(x_flat, u_true_cs, 'k--', label='True')
        plt.title(f't={t_cs}')
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        plt.legend()

        print(f"True min: {u_true.min()}, True max: {u_true.max()}")