import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as opt

def train(count, nor_count, lib_size, network, output_dir=None, optimizer='rmsprop', learning_rate=None,
          epochs=300, reduce_lr=10, output_subset=None, use_raw_as_output=True,
          early_stop=15, batch_size=256, clip_grad=5., save_weights=False,
          validation_split=0.1, tensorboard=False, verbose=True, ae_type='normal',
          warm_up = 0, Z_fixed_idx = None,
          **kwds):

    model = network.model
    loss = network.loss
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.RMSprop(clipvalue=clip_grad)
    else:
        optimizer = opt.RMSprop(lr=learning_rate, clipvalue=clip_grad)


    train_dataset = tf.data.Dataset.from_tensor_slices((count, nor_count, lib_size))
    train_dataset = train_dataset.shuffle(buffer_size=count.shape[0],
                                          reshuffle_each_iteration=True).batch(batch_size)

    @tf.function
    def train_step_warm_up(count, nor_count, lib_size):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            outputs = model([nor_count, lib_size], training=True)
            # Compute the loss value for this minibatch.
            if ae_type == 'normal':
                loss_value = loss(nor_count, outputs)
            else:
                loss_value = loss(count, outputs)
            # Add archetypal loss term
            loss_value += sum(model.losses)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        var_list = model.trainable_weights[:Z_fixed_idx] + model.trainable_weights[(Z_fixed_idx+1):]
        grads = tape.gradient(loss_value, var_list)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, var_list))
        
        return loss_value
    
    @tf.function
    def train_step(count, nor_count, lib_size):
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            outputs = model([nor_count, lib_size], training=True)
            # Compute the loss value for this minibatch.
            if ae_type == 'normal':
                loss_value = loss(nor_count, outputs)
            else:
                loss_value = loss(count, outputs)
            # Add archetypal loss term
            loss_value += sum(model.losses)
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
 
        return loss_value

    best_weights = None
    wait_early_stop = wait_reduce_lr = 0
    stopped_epoch = 0
    best = np.Inf
    factor = 0.5
    min_lr = 0.001

    for epoch in range(epochs):

        # Iterate over the batches of the dataset.
        for step, (count, nor_count, lib_size) in enumerate(train_dataset):
            #print('Step: ', step)
            if epoch < warm_up:
                loss_value = train_step_warm_up(count, nor_count, lib_size)
            else:
                loss_value = train_step(count, nor_count, lib_size)
        
        current = loss_value
        
        if epoch % 50 == 0:
            print("Training loss at epoch %d (for the last batch): %.4f"
                  % (epoch, float(loss_value)))
            print("Learning rate is: %.6f" % float(tf.keras.backend.get_value(optimizer.lr)))
            print(model.trainable_weights[Z_fixed_idx])
                
        if np.less(current, best):
            best = current
            wait_early_stop = wait_reduce_lr = 0
            best_weights = model.get_weights()
        else:
            wait_early_stop += 1
            wait_reduce_lr += 1
            if wait_early_stop >= early_stop:
                stopped_epoch = epoch
                print("Restoring model weights from the end of the best epoch.")
                model.set_weights(best_weights)
            if wait_reduce_lr >= reduce_lr:
                wait_reduce_lr = 0
                lr = float(tf.keras.backend.get_value(optimizer.lr))
                if lr*factor > min_lr:
                    tf.keras.backend.set_value(optimizer.lr, lr*factor)
                else:
                    tf.keras.backend.set_value(optimizer.lr, min_lr)
                
        if stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (stopped_epoch + 1))
            break
            
    return None
