"""
Script for training Octo model on custom robotics data.
Handles multi-session training data in JSONL format with accompanying image directories.

Dataset structure:
training_data/
    session_YYYYMMDD_HHMMSS/
        data.jsonl
        main_images/
        wrist_images/
"""
from absl import flags
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from tqdm import tqdm
import os
import json
import glob
import numpy as np
from typing import List, Dict
from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_utils import freeze_weights, process_text, TrainState

# Command line arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", "hf://rail-berkeley/octo-base-1.5", "Path to pre-trained Octo checkpoint")
flags.DEFINE_string("data_dir", "/home/scferro/Documents/final_project/training_data_vla", "Path to directory containing all training sessions")
flags.DEFINE_string("save_dir", "/home/scferro/Documents/final_project/octo_models", "Directory for saving checkpoints")
flags.DEFINE_integer("batch_size", 2, "Physical batch size for training")  # Set back to 4 for now
flags.DEFINE_integer("gradient_accumulation_steps", 64, "Number of gradient accumulation steps")  # Add this line
flags.DEFINE_bool("freeze_transformer", False, "Whether to freeze pre-trained transformer weights")

def load_session_data(session_dir: str, max_horizon: int = 4, sample_freq: int = 2) -> List[Dict]:
    """
    Load data and create sequences where each sequence has:
    - Two frames of images (t-2 and t)
    - Four consecutive actions (t, t+1, t+2, t+3)
    """
    jsonl_path = os.path.join(session_dir, "data.jsonl")
    sequences = []
    full_sequence = []
    
    # Load all data first
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data['main_image'] = os.path.join(session_dir, data['main_image'])
            data['wrist_image'] = os.path.join(session_dir, data['wrist_image'])
            data['language_instruction'] = data.pop('prompt')
            full_sequence.append(data)
    
    # Create sequences
    for current_idx in range(sample_freq, len(full_sequence) - (max_horizon - 1)):
        prev_frame_idx = current_idx - sample_freq
        
        sequence = {
            'current_frame': full_sequence[current_idx],
            'prev_frame': full_sequence[prev_frame_idx],
            'actions': [full_sequence[i]['action'] 
                       for i in range(current_idx, current_idx + max_horizon)],
        }
        sequences.append(sequence)
    
    print(f"Created {len(sequences)} sequences from {session_dir}")
    print(f"Each sequence has 2 frames ({sample_freq} steps apart) and {max_horizon} consecutive actions")
    
    return sequences

def create_tf_dataset(sequences: List[Dict], batch_size: int, max_horizon: int = 4):
    """Convert sequences to TensorFlow dataset format for Octo."""
    
    def load_image(path, size=(256, 256)):
        img_data = tf.io.read_file(path)
        img = tf.io.decode_png(img_data, channels=3)
        return tf.cast(tf.image.resize(img, size), tf.float32) / 255.0
    
    def process_sequence(sequence):
        """Process a single sequence into the format Octo expects."""
        # Create image tensors
        current_frame = sequence['current_frame']
        prev_frame = sequence['prev_frame']
        
        # Stack images temporally with correct sizes
        main_images = tf.stack([
            load_image(prev_frame['main_image'], size=(256, 256)),
            load_image(current_frame['main_image'], size=(256, 256))
        ])
        wrist_images = tf.stack([
            load_image(prev_frame['wrist_image'], size=(128, 128)),
            load_image(current_frame['wrist_image'], size=(128, 128))
        ])
        
        # Convert actions and reshape to match expected format
        actions = tf.convert_to_tensor(sequence['actions'], dtype=tf.float32)  # Shape (4, 7)
        # Reshape to (2, 4, 7) - duplicating the actions for both timesteps
        actions = tf.stack([actions, actions])  # This creates shape (2, 4, 7)
        
        # Create action pad mask that matches the action shape (2, 4, 7)
        action_pad_mask = tf.ones((2, 4, 7), dtype=tf.bool)  # Create (window, horizon, dim) mask
        
        obs = {
            'image_primary': main_images,
            'image_wrist': wrist_images,
            'pad_mask_dict': {
                'image_primary': tf.ones([2], dtype=tf.bool),
                'image_wrist': tf.ones([2], dtype=tf.bool)
            },
            'timestep': tf.range(2, dtype=tf.int32),
            'timestep_pad_mask': tf.ones([2], dtype=tf.bool),
        }
        
        return {
            'language_instruction': current_frame['language_instruction'],
            'observation': obs,
            'action': actions,  # Shape (2, 4, 7)
            'action_pad_mask': action_pad_mask,  # Shape (2, 4, 7)
            'done': tf.zeros([max_horizon], dtype=tf.bool),
        }
    
    # Define specs for observation with correct image sizes
    output_signature = {
        'language_instruction': tf.TensorSpec(shape=(), dtype=tf.string),
        'observation': {
            'image_primary': tf.TensorSpec(shape=(2, 256, 256, 3), dtype=tf.float32),
            'image_wrist': tf.TensorSpec(shape=(2, 128, 128, 3), dtype=tf.float32),
            'pad_mask_dict': {
                'image_primary': tf.TensorSpec(shape=(2,), dtype=tf.bool),
                'image_wrist': tf.TensorSpec(shape=(2,), dtype=tf.bool)
            },
            'timestep': tf.TensorSpec(shape=(2,), dtype=tf.int32),
            'timestep_pad_mask': tf.TensorSpec(shape=(2,), dtype=tf.bool),
        },
        'action': tf.TensorSpec(shape=(2, max_horizon, 7), dtype=tf.float32),
        'action_pad_mask': tf.TensorSpec(shape=(2, max_horizon, 7), dtype=tf.bool),
        'done': tf.TensorSpec(shape=(max_horizon,), dtype=tf.bool),
    }

    def generate_sequences():
        for seq in sequences:
            yield process_sequence(seq)
    
    dataset = tf.data.Dataset.from_generator(
        generate_sequences,
        output_signature=output_signature
    )
    
    dataset = dataset.shuffle(1800, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    
    return dataset

def process_batch_text(batch, text_processor):
    """Process text instructions in the batch."""
    instructions = [x.numpy().decode('utf-8') for x in batch['language_instruction']]
    
    tokenized = text_processor.tokenizer(
        instructions,
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='np'
    )
    
    processed = {
        'input_ids': tf.convert_to_tensor(tokenized['input_ids'], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(tokenized['attention_mask'], dtype=tf.int32)
    }
    
    batch = dict(batch)
    batch['task'] = {'language_instruction': processed}
    del batch['language_instruction']
    
    return batch

def convert_batch_to_jax(batch):
    """Convert TensorFlow batch to JAX arrays."""
    def _convert(x):
        if isinstance(x, tf.Tensor):
            return jnp.array(x.numpy())
        elif isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            if all(isinstance(item, dict) for item in x):
                return [_convert(item) for item in x]
            return jnp.array([_convert(item) for item in x])
        return x
    
    return _convert(batch)

def create_train_state(model):
    """Initialize training state with optimizer and accumulated gradients."""
    learning_rate = optax.join_schedules(
        [
            # optax.linear_schedule(0, 3e-7, 100),
            optax.constant_schedule(3e-7)
        ],
        [100]
    )
    
    optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Keep original frozen keys
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    optimizer = freeze_weights(optimizer, model.params, frozen_keys)
    
    # Initialize accumulated gradients with zeros like model params
    acc_grads = jax.tree_map(lambda x: jnp.zeros_like(x), model.params)
    
    # Create state with additional accumulated gradients field
    state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=optimizer,
    )
    
    return state, acc_grads

def create_loss_fn(model):
    """Create loss function with access to model."""
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics
    return loss_fn

def create_train_step(model):
    """Create training step with memory-efficient gradient accumulation."""
    loss_fn = create_loss_fn(model)

    @jax.jit
    def train_step(state: TrainState, acc_grads, batch, step_idx):
        rng, dropout_rng = jax.random.split(state.rng)
        
        # Compute gradients for current batch
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        
        # Scale both loss and gradients
        scaled_loss = loss / FLAGS.gradient_accumulation_steps
        scaled_grads = jax.tree_map(lambda g: g / FLAGS.gradient_accumulation_steps, grads)
        
        # Add to accumulated gradients
        acc_grads = jax.tree_map(
            lambda acc, g: acc + g,
            acc_grads, scaled_grads
        )
        
        # Compute whether we should update without using Python control flow
        should_update = (step_idx + 1) % FLAGS.gradient_accumulation_steps == 0
        
        # Create two possible next states using tree_map
        updated_state = state.apply_gradients(grads=acc_grads, rng=rng)
        acc_grads_reset = jax.tree_map(lambda x: jnp.zeros_like(x), acc_grads)
        acc_grads_keep = acc_grads
        
        # Use where to select between states without Python conditionals
        new_state = jax.tree_map(
            lambda x, y: jnp.where(should_update, x, y),
            updated_state, state
        )
        new_acc_grads = jax.tree_map(
            lambda x, y: jnp.where(should_update, x, y),
            acc_grads_reset, acc_grads_keep
        )
            
        return new_state, new_acc_grads, metrics, scaled_loss
    
    return train_step

def main(_):
    assert FLAGS.batch_size % jax.device_count() == 0, "Batch size must be divisible by device count"
    
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")
    
    print("Loading pre-trained model...")
    model = OctoModel.load_pretrained(FLAGS.pretrained_path)
    
    max_horizon = 4  # Keep this as it matches original model
    sample_freq = 2  # Keep this as it matches original data format
    
    # Load all training sessions
    print("Loading training data...")
    all_sequences = []
    session_dirs = glob.glob(os.path.join(FLAGS.data_dir, "session_*"))
    print(f"Found {len(session_dirs)} session directories")
    for session_dir in session_dirs:
        sequences = load_session_data(session_dir, max_horizon=max_horizon, sample_freq=sample_freq)
        all_sequences.extend(sequences)
    print(f"Created total of {len(all_sequences)} sequences")
    
    # Create dataset before train state
    train_dataset = create_tf_dataset(all_sequences, FLAGS.batch_size, max_horizon=max_horizon)
    
    # Create train state with accumulated gradients
    train_state, acc_grads = create_train_state(model)
    train_step = create_train_step(model)
    
    print("\nStarting training...")
    print(f"Using batch size {FLAGS.batch_size} with {FLAGS.gradient_accumulation_steps} accumulation steps")
    print(f"Effective batch size: {FLAGS.batch_size * FLAGS.gradient_accumulation_steps}")
    
    train_iter = iter(train_dataset)
    running_loss = 0.0
    batch_losses = []  # Store individual batch losses
    steps = 5000 * 128 / FLAGS.batch_size

    try:
        with tqdm(range(int(steps)), desc="Training") as pbar:
            for step in pbar:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataset)
                    batch = next(train_iter)
                
                batch = process_batch_text(batch, model.text_processor)
                jax_batch = convert_batch_to_jax(batch)
                
                train_state, acc_grads, metrics, loss = train_step(
                    train_state, acc_grads, jax_batch, step
                )
                
                batch_losses.append(float(loss))  # Store this batch's loss
                
                if (step + 1) % FLAGS.gradient_accumulation_steps == 0:
                    # Calculate average loss for this accumulated batch
                    avg_loss = sum(batch_losses) / len(batch_losses)
                    print(f"\nStep {step+1} (Gradient Update):")
                    print(f"Average loss for accumulated batch: {avg_loss:.4f}")
                    running_loss = 0.9 * running_loss + 0.1 * avg_loss
                    batch_losses = []  # Reset for next accumulation
                
                pbar.set_postfix({
                    'running_loss': f"{running_loss:.4f}",
                    'curr_batch_loss': f"{loss:.4f}",
                    'step': step + 1,
                    'acc_step': (step + 1) % FLAGS.gradient_accumulation_steps
                })
                
                # More frequent checkpoints early in training
                if step < 1000 and (step + 1) % 100 == 0:
                    print(f"\nEarly checkpoint at step {step + 1}")
                    train_state.model.save_pretrained(
                        step=step,
                        checkpoint_path=FLAGS.save_dir
                    )
                elif (step + 1) % 1000 == 0:
                    print(f"\nSaving checkpoint at step {step + 1}")
                    train_state.model.save_pretrained(
                        step=step,
                        checkpoint_path=FLAGS.save_dir
                    )
                    
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Saving checkpoint due to error...")
        train_state.model.save_pretrained(
            step=step,
            checkpoint_path=FLAGS.save_dir
        )
        raise e

if __name__ == "__main__":
    flags.FLAGS([""])  # Initialize flags
    main(None)