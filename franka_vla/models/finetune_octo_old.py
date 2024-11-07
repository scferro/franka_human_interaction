from absl import app, flags, logging
import flax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tqdm
import wandb
import os
import json
import glob
import numpy as np
from typing import List, Dict
from functools import partial
import gc
import jax.numpy as jnp

from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_utils import process_text, TrainState, create_optimizer

FLAGS = flags.FLAGS

# Define required flags
flags.DEFINE_string("pretrained_path", "hf://rail-berkeley/octo-base-1.5", "Path to pre-trained Octo checkpoint")
flags.DEFINE_string("data_dir", "/home/scferro/training_data_vla", "Path to directory containing all training sessions")
flags.DEFINE_string("save_dir", "/home/scferro/checkpoints", "Directory for saving checkpoints")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_integer("steps", 20000, "Steps for training")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate for training")

class ActionProcessor:
    def __init__(self):
        self.stats = None
        
    def compute_statistics(self, all_sequences):
        actions = []
        for seq in all_sequences:
            for action in seq['actions']:
                actions.append(action)
        actions = np.array(actions)
        continuous = actions[:, :6]
        
        self.stats = {
            'mean': np.mean(continuous, axis=0),
            'std': np.std(continuous, axis=0)
        }
        return self.stats
    
    def normalize_actions(self, actions):
        continuous = actions[:, :6]
        discrete = actions[:, 6:]
        normalized = (continuous - self.stats['mean']) / self.stats['std']
        return np.concatenate([normalized, discrete], axis=1)
    
    def save_statistics(self, save_dir):
        stats_path = os.path.join(save_dir, 'action_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'mean': self.stats['mean'].tolist(),
                'std': self.stats['std'].tolist()
            }, f)

def load_raw_session_data(session_dir: str, max_horizon: int = 4, sample_freq: int = 2) -> List[Dict]:
    """Load raw session data without normalization"""
    jsonl_path = os.path.join(session_dir, "data.jsonl")
    sequences = []
    full_sequence = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data['main_image'] = os.path.join(session_dir, data['main_image'])
            data['wrist_image'] = os.path.join(session_dir, data['wrist_image'])
            data['language_instruction'] = data.pop('prompt')
            full_sequence.append(data)
    
    for current_idx in range(sample_freq, len(full_sequence) - (max_horizon - 1)):
        actions = [full_sequence[i]['action'] 
                  for i in range(current_idx, current_idx + max_horizon)]
        
        sequence = {
            'current_frame': full_sequence[current_idx],
            'prev_frame': full_sequence[current_idx - sample_freq],
            'actions': actions,
        }
        sequences.append(sequence)
    
    return sequences

def load_session_data(session_dir: str, action_processor, max_horizon: int = 4, sample_freq: int = 2) -> List[Dict]:
    """Load and normalize session data"""
    sequences = load_raw_session_data(session_dir, max_horizon, sample_freq)
    
    if action_processor is not None:
        for sequence in sequences:
            actions = np.array(sequence['actions'])
            sequence['actions'] = action_processor.normalize_actions(actions)
    
    return sequences

def create_tf_dataset(sequences: List[Dict], batch_size: int, max_horizon: int = 4):
    """Convert sequences to TensorFlow dataset format for Octo."""
    
    def load_image(path, size=(256, 256)):
        img_data = tf.io.read_file(path)
        img = tf.io.decode_png(img_data, channels=3)
        return tf.cast(tf.image.resize(img, size), tf.float32) / 255.0
    
    def process_sequence(sequence):
        """Process a single sequence into Octo's expected format."""
        current_frame = sequence['current_frame']
        prev_frame = sequence['prev_frame']
        
        # Load images for observation sequence
        main_images_obs = tf.stack([
            load_image(prev_frame['main_image'], size=(256, 256)),
            load_image(current_frame['main_image'], size=(256, 256))
        ])
        wrist_images_obs = tf.stack([
            load_image(prev_frame['wrist_image'], size=(128, 128)),
            load_image(current_frame['wrist_image'], size=(128, 128))
        ])
        
        # For task, use only the current frame
        current_main = load_image(current_frame['main_image'], size=(256, 256))
        current_wrist = load_image(current_frame['wrist_image'], size=(128, 128))
        
        # Create proper masks
        timestep_mask = tf.ones([2], dtype=tf.bool)
        action_pad_mask = tf.ones((2, 4, 7), dtype=tf.bool)
        
        # Convert actions
        actions = tf.convert_to_tensor(sequence['actions'], dtype=tf.float32)
        actions = tf.stack([actions, actions])
        
        return {
            'observation': {
                'image_primary': main_images_obs,
                'image_wrist': wrist_images_obs,
                'pad_mask_dict': {
                    'image_primary': timestep_mask,
                    'image_wrist': timestep_mask
                },
                'timestep': tf.range(2, dtype=tf.int32),
                'timestep_pad_mask': timestep_mask,
            },
            'task': {
                'language_instruction': tf.convert_to_tensor(current_frame['language_instruction']),
                'image_primary': current_main,
                'image_wrist': current_wrist
            },
            'action': actions,
            'action_pad_mask': action_pad_mask,
            'done': tf.zeros([max_horizon], dtype=tf.bool),
        }
    
    dataset = tf.data.Dataset.from_generator(
        lambda: (process_sequence(s) for s in sequences),
        output_signature={
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
            'task': {
                'language_instruction': tf.TensorSpec(shape=(), dtype=tf.string),
                'image_primary': tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
                'image_wrist': tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32)
            },
            'action': tf.TensorSpec(shape=(2, max_horizon, 7), dtype=tf.float32),
            'action_pad_mask': tf.TensorSpec(shape=(2, max_horizon, 7), dtype=tf.bool),
            'done': tf.TensorSpec(shape=(max_horizon,), dtype=tf.bool),
        }
    )
    
    dataset = dataset.shuffle(1800, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_train_state(model):
    """Initialize training state with optimizer."""
    lr = FLAGS.learning_rate
    
    frozen_keys = [
        "octo_transformer.*",
        "task_preprocessor.*", 
        "observation_preprocessor.*"
    ]

    # Create flat mask for parameters
    flat_params = flax.traverse_util.flatten_dict(model.params)
    trainable = {k: not any(pattern in k for pattern in frozen_keys) 
                for k in flat_params.keys()}
    
    # Create the optimizers
    tx = optax.chain(
        optax.masked(
            optax.sgd(learning_rate=lr),
            mask=flax.traverse_util.unflatten_dict(trainable)
        )
    )
    
    state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )
    
    return state

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

def convert_batch_to_jax(batch):
    """Convert TensorFlow batch to JAX arrays."""
    def _convert(x):
        if isinstance(x, tf.Tensor):
            return jnp.array(x.numpy())
        elif isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return jnp.array([_convert(item) for item in x])
        return x
    return _convert(batch)

def main(_):
    if not os.path.exists(FLAGS.data_dir):
        raise ValueError(f"Data directory does not exist: {FLAGS.data_dir}")
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        logging.info(f"Created save directory: {FLAGS.save_dir}")
    
    assert FLAGS.batch_size % jax.device_count() == 0, "Batch size must be divisible by device count"
    
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")
    
    # Setup wandb
    wandb.init(project="octo_training")
    
    # Load pre-trained model
    logging.info("Loading pre-trained model...")
    model = OctoModel.load_pretrained(FLAGS.pretrained_path)
    
    # Initialize action processor
    action_processor = ActionProcessor()
    
    # Load raw sequences first to compute statistics
    raw_sequences = []
    session_dirs = glob.glob(os.path.join(FLAGS.data_dir, "session_*"))
    for session_dir in session_dirs:
        sequences = load_raw_session_data(session_dir)
        raw_sequences.extend(sequences)

    # Compute and save statistics
    stats = action_processor.compute_statistics(raw_sequences)
    action_processor.save_statistics(FLAGS.save_dir)

    # Update model's statistics for inference
    model.dataset_statistics["bridge_dataset"]["action"] = stats

    # Now load normalized sequences
    all_sequences = []
    for session_dir in session_dirs:
        sequences = load_session_data(session_dir, action_processor)
        all_sequences.extend(sequences)
    
    # Create dataset and iterator
    train_dataset = create_tf_dataset(all_sequences, FLAGS.batch_size)
    train_iter = iter(train_dataset)
    
    # Create train state
    train_state = create_train_state(model)
    loss_fn = create_loss_fn(model)
    
    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        del grads
        return new_state, info, loss  # Return loss explicitly
    
    # Initialize running average
    ema_loss = None
    ema_alpha = 0.98  # Smoothing factor: higher = smoother
    
    logging.info("Starting training...")
    steps = FLAGS.steps
    with tqdm.tqdm(total=steps, dynamic_ncols=True) as pbar:
        for i in range(steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                batch = next(train_iter)
                gc.collect()
            
            # Process batch
            if isinstance(batch['task']['language_instruction'], tf.Tensor):
                batch['task']['language_instruction'] = batch['task']['language_instruction'].numpy()
            batch = process_text(batch, model.text_processor)
            jax_batch = convert_batch_to_jax(batch)
            del batch
            
            # Single training step
            train_state, update_info, batch_loss = train_step(train_state, jax_batch)
            del jax_batch
            
            # Update running average
            batch_loss = float(batch_loss)  # Convert to Python float
            if ema_loss is None:
                ema_loss = batch_loss
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * batch_loss
                
            # Logging
            if (i + 1) % 50 == 0:
                try:
                    # Only log loss values
                    wandb.log({
                        "loss/batch": float(batch_loss),
                        "loss/smooth": float(ema_loss),
                    }, step=i)
                except Exception as e:
                    logging.warning(f"Logging failed: {e}")
            
            # Update progress bar with smoothed loss
            pbar.set_postfix({
                'smooth_loss': f"{ema_loss:.4f}",
                'batch_loss': f"{batch_loss:.4f}",
                'step': i + 1
            })
            pbar.update(1)  # Manually update progress
            
            # Checkpointing
            if (i + 1) % 1000 == 0:
                train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)
                gc.collect()
                jax.clear_caches()

if __name__ == "__main__":
    flags.FLAGS([""])  # Initialize flags
    app.run(main)