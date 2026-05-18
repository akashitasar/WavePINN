import scipy.optimize
import numpy as np
import tensorflow as tf

class L_BFGS_B:
    """
    Optimize the keras network model using L-BFGS-B algorithm.

    Attributes:
        model: optimization target model.
        samples: training samples.
        factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
               1e7 for moderate accuracy; 10 for extremely high accuracy.
        m: maximum number of variable metric corrections used to define the limited memory matrix.
        maxls: maximum number of line search steps (per iteration).
        maxiter: maximum number of iterations.
        metris: logging metrics.
        progbar: progress bar.
    """

    def __init__(self, model, x_train, y_train, factr=1e5, m=50, maxls=50, maxiter=15000):
        """
        Args:
            model: optimization target model.
            samples: training samples.
            factr: convergence condition. typical values for factr are: 1e12 for low accuracy;
                   1e7 for moderate accuracy; 10.0 for extremely high accuracy.
            m: maximum number of variable metric corrections used to define the limited memory matrix.
            maxls: maximum number of line search steps (per iteration).
            maxiter: maximum number of iterations.
        """

        # set attributes
        self.model = model
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.metrics = ['loss']
        # initialize the progress bar
        self.progbar = tf.keras.callbacks.ProgbarLogger(
            count_mode='steps', stateful_metrics=self.metrics)
        self.progbar.set_params( {
            'verbose':1, 'epochs':1, 'steps':self.maxiter, 'metrics':self.metrics})
        # store iteration-based losses
        self.iter_loss_history = []
        self.iter_loss_eqn = []
        self.iter_loss_ini = []
        self.iter_loss_dini = []
        self.iter_loss_bnd = []
        self.iteration = 0
    def set_weights(self, flat_weights):
        """
        Set weights to the model.

        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        """
        Evaluate loss and gradients for weights as tf.Tensor.

        Args:
            x: input data.

        Returns:
            loss and gradients for weights as tf.Tensor.
        """

        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)

        # forward pass ONCE
        y_pred = self.model(self.x_train)

        y_eqn, y_ini, y_dini, y_bnd = y_pred

        # individual losses
        loss_eqn = tf.reduce_mean(tf.square(y_eqn - self.y_train[0]))
        loss_ini = tf.reduce_mean(tf.square(y_ini - self.y_train[1]))
        loss_dini = tf.reduce_mean(tf.square(y_dini - self.y_train[2]))
        loss_bnd = tf.reduce_mean(tf.square(y_bnd - self.y_train[3]))

        loss = loss_eqn + loss_ini + loss_dini + loss_bnd

        # store convergence (NO extra compute later)
        self.iter_loss_history.append(loss.numpy())
        self.iter_loss_eqn.append(loss_eqn.numpy())
        self.iter_loss_ini.append(loss_ini.numpy())
        self.iter_loss_dini.append(loss_dini.numpy())
        self.iter_loss_bnd.append(loss_bnd.numpy())

        # gradient computation
        with tf.GradientTape() as g:
            y_pred = self.model(self.x_train)
            y_eqn, y_ini, y_dini, y_bnd = y_pred

            loss = (
                    tf.reduce_mean(tf.square(y_eqn - self.y_train[0])) +
                    tf.reduce_mean(tf.square(y_ini - self.y_train[1])) +
                    tf.reduce_mean(tf.square(y_dini - self.y_train[2])) +
                    tf.reduce_mean(tf.square(y_bnd - self.y_train[3]))
            )

        grads = g.gradient(loss, self.model.trainable_variables)

        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        loss = loss.numpy().astype('float64')

        return loss, grads

    def callback(self, weights):
        """
        Callback that runs once per L-BFGS iteration.
        ONLY handles logging (no model calls).
        """

        self.iteration += 1

        # progress bar only (no recomputation)
        self.progbar.on_batch_begin(0)
        self.progbar.on_batch_end(
            0,
            logs=dict(zip(self.metrics, [self.iter_loss_history[-1]]))
        )

    def fit(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.model.get_weights() ])
        # optimize the weight vector
        print('Optimizer: L-BFGS-B (maxiter={})'.format(self.maxiter))
        self.progbar.on_train_begin()
        self.progbar.on_epoch_begin(1)
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
            factr=self.factr, m=self.m, maxls=self.maxls, maxiter=self.maxiter,
            callback=self.callback)
        self.progbar.on_epoch_end(1)
        self.progbar.on_train_end()