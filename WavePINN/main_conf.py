import star.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from star.pinn import PINN
from star.network import Network
from star.optimizer import L_BFGS_B
import time

def u0(tx, c=1, k=2, sd=0.5):
    """
    Initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.
        c: wave velocity.
        k: wave number.
        sd: standard deviation.

    Returns:
        u(t, x) as tf.Tensor.
    """

    t = tx[..., 0, None]
    x = tx[..., 1, None]
    z = k*x - (c*k)*t
    return tf.sin(z) * tf.exp(-(0.5*z/sd)**2)

def du0_dt(tx):
    """
    First derivative of t for the initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.

    Returns:
        du(t, x)/dt as tf.Tensor.
    """

    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 1000

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network).build()

    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = 4*tx_eqn[..., 0]                # t =  0 ~ +4
    tx_eqn[..., 1] = tx_eqn[..., 1]             # x = 0 ~ +1
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                               # t = 0
    tx_ini[..., 1] = tx_ini[..., 1]             # x = 0 ~ +1
    N = num_train_samples  # 10000
    Nb = N // 2  # 5000
    tx_bnd_left = np.random.rand(Nb, 2)
    tx_bnd_left[..., 0] = 4 * tx_bnd_left[..., 0]
    tx_bnd_left[..., 1] = 0

    tx_bnd_right = np.random.rand(Nb, 2)
    tx_bnd_right[..., 0] = 4 * tx_bnd_right[..., 0]
    tx_bnd_right[..., 1] = 1
    t_left = tx_bnd_left[..., 0:1]       #create sine impulse at left boundary

    u_left = np.sin(np.pi * t_left / 0.2)
    u_left[t_left > 0.2] = 0.0

    u_right = np.zeros((Nb, 1))   #right boundary =0

    tx_bnd = np.vstack([tx_bnd_left, tx_bnd_right])    #combine boundary data
    u_bnd = np.vstack([u_left, u_right])
    # create training output (trained output uses these to learn physics)
    u_zero = np.zeros((num_train_samples, 1))
    u_ini = np.zeros((num_train_samples, 1))
    du_dt_ini = np.zeros((num_train_samples, 1))

    print(tx_eqn.shape)
    print(tx_ini.shape)
    print(tx_bnd.shape)
    print(u_zero.shape)
    print(u_ini.shape)
    print(du_dt_ini.shape)
    print(u_bnd.shape)

    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_zero, u_ini, du_dt_ini, u_bnd]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    start_time = time.perf_counter()
    lbfgs.fit()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print(f"\nTraining time: {minutes} min {seconds:.2f} sec")

    plt.figure(figsize=(8, 5))

    plt.plot(lbfgs.iter_loss_eqn, label='PDE loss')
    plt.plot(lbfgs.iter_loss_ini, label='IC loss')
    plt.plot(lbfgs.iter_loss_dini, label='IC velocity loss')
    plt.plot(lbfgs.iter_loss_bnd, label='Boundary loss')

    plt.yscale('log')
    plt.xlabel('Iteration')  # ✅ TRUE iterations now
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs L-BFGS Iterations')

    plt.show()

    # predict u(t,x) distribution
    t_flat = np.linspace(0, 4, num_test_samples)
    x_flat = np.linspace(0, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    vmin, vmax = -0.5, +0.5
    plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(vmin, vmax)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [1, 2, 3]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig('result_img_neumann.png', transparent=True)
    plt.show()