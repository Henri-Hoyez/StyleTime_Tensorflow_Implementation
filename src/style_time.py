import tensorflow as tf
from src.models import losses
from src.utils.gpu_memory_grow import gpu_memory_grow



# StyleTime Algorithm
def style_time(content_series, style_series, iterations=5000, alpha=0.5 , beta=1000.0, gamma=0.0001, learning_rate=0.01):
    """Implements the StyleTime algorithm."""
    tf.keras.layers.Dense(100)
    
    # Convert the input series to tensors
    y = tf.Variable(content_series, dtype=tf.float32)
    
    c_losses = []
    s_losses = []
    total_losses = []
    
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            # Compute total loss
            c_loss = losses.content_loss(y, content_series)
            s_loss = losses.style_loss(y, style_series)
            tv_loss = losses.total_variation_loss(y)
            
            total_loss = alpha * c_loss + beta * s_loss + gamma * tv_loss
            
        c_losses.append(c_loss.numpy())
        s_losses.append(s_loss.numpy())
        total_losses.append(total_loss.numpy())
            
        # Compute gradients and apply optimization
        gradients = tape.gradient(total_loss, [y])
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, [y]))
        
        # if i % 10 == 0:
        #     print(f"\rIteration {i}/{iterations}. content loss {alpha* c_loss:0.2f}, Style loss {beta* s_loss:0.2f}, tv loss {gamma * tv_loss:0.2f}, Total Loss: {total_loss.numpy():0.2f}", end="")
        # exit()

    return y.numpy(), c_losses, s_losses, total_losses