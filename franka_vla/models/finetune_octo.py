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
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.train_utils import process_text, TrainState, create_optimizer
from octo.utils.spec import ModuleSpec

FLAGS = flags.FLAGS

# Define required flags
flags.DEFINE_string("pretrained_path", "hf://rail-berkeley/octo-base-1.5", "Path to pre-trained Octo checkpoint")
flags.DEFINE_string("data_dir", "/home/scferro/training_data_vla", "Path to directory containing all training sessions")
flags.DEFINE_string("save_dir", "/home/scferro/checkpoints", "Directory for saving checkpoints")
flags.DEFINE_integer("batch_size", 32, "Batch size for training")
flags.DEFINE_integer("steps", 20000, "Steps for training")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate for training")
flags.DEFINE_bool("freeze_transformer", False, "Whether pre-trained transformer weights should be frozen.",)

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

def create_square_crop(image: tf.Tensor, justify: str = 'center') -> tf.Tensor:
    """Create a square crop of the image with specified justification."""
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    
    crop_size = tf.minimum(height, width)
    
    if justify == 'center':
        start_x = (width - crop_size) // 2
    elif justify == 'left':
        start_x = 0
    elif justify == 'right':
        start_x = width - crop_size
    else:
        raise ValueError(f"Unknown justification: {justify}")
        
    start_y = (height - crop_size) // 2
    
    return tf.image.crop_to_bounding_box(
        image, start_y, start_x, crop_size, crop_size
    )

def augment_image(image: tf.Tensor, seed: int) -> tf.Tensor:
    """Apply random brightness and contrast augmentation."""
    # Use consistent seed for each image
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.1, seed=(seed, 0)
    )
    image = tf.image.stateless_random_contrast(
        image, lower=0.9, upper=1.1, seed=(seed, 1)
    )
    # Ensure values stay in valid range
    return tf.clip_by_value(image, 0.0, 1.0)

def load_and_process_image(path: str, size: tuple, justify: str = 'center', 
                         augment: bool = False, aug_seed: int = None) -> tf.Tensor:
    """Load, crop, resize, and optionally augment an image."""
    img_data = tf.io.read_file(path)
    img = tf.io.decode_png(img_data, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Square crop with specified justification
    img = create_square_crop(img, justify)
    
    # Resize to target size
    img = tf.image.resize(img, size)
    
    # Apply augmentation if requested
    if augment and aug_seed is not None:
        img = augment_image(img, aug_seed)
        
    return img

def create_augmented_sequences(sequence: Dict, justify_options: List[str] = ['center', 'left', 'right']) -> List[Dict]:
    """Create multiple sequences with different crops and augmentations."""
    augmented_sequences = []
    
    # Create sequences with different crops
    for justify in justify_options:
        # Create base sequence with specified crop
        crop_sequence = {
            'current_frame': sequence['current_frame'].copy(),
            'prev_frame': sequence['prev_frame'].copy(),
            'actions': sequence['actions'].copy(),
            'proprioception': sequence['proprioception'].copy(),
            'justify_main': justify,
            'augment': False,
            'aug_seed_main': None,
            'aug_seed_wrist': None,
        }
        augmented_sequences.append(crop_sequence)
        
        # Create augmented version of the sequence
        aug_sequence = {
            'current_frame': sequence['current_frame'].copy(),
            'prev_frame': sequence['prev_frame'].copy(),
            'actions': sequence['actions'].copy(),
            'proprioception': sequence['proprioception'].copy(),
            'justify_main': justify,
            'augment': True,
            'aug_seed_main': np.random.randint(0, 2**31),
            'aug_seed_wrist': np.random.randint(0, 2**31),
        }
        augmented_sequences.append(aug_sequence)
    
    return augmented_sequences

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
        proprio = [full_sequence[i]['proprioception']
                  for i in range(current_idx, current_idx + max_horizon)]
        
        sequence = {
            'current_frame': full_sequence[current_idx],
            'prev_frame': full_sequence[current_idx - sample_freq],
            'actions': actions,
            'proprioception': proprio,
        }
        sequences.append(sequence)
    
    return sequences

def load_session_data(session_dir: str, action_processor, max_horizon: int = 4, sample_freq: int = 2) -> List[Dict]:
    """Load and normalize session data with augmentations"""
    base_sequences = load_raw_session_data(session_dir, max_horizon, sample_freq)
    
    # Create augmented sequences
    augmented_sequences = []
    for seq in base_sequences:
        if action_processor is not None:
            actions = np.array(seq['actions'])
            seq['actions'] = action_processor.normalize_actions(actions)
            seq['proprioception'] = np.array(seq['proprioception'])
        
        # Create variations with different crops and augmentations
        augmented_sequences.extend(create_augmented_sequences(seq))
    
    return augmented_sequences

def create_tf_dataset(sequences: List[Dict], batch_size: int, max_horizon: int = 4):
    """Convert sequences to TensorFlow dataset format with augmentations."""
    
    def process_sequence(sequence):
        """Process a single sequence into Octo's expected format."""
        current_frame = sequence['current_frame']
        prev_frame = sequence['prev_frame']
        
        # Load images for observation sequence with specified crops and augmentations
        main_images_obs = tf.stack([
            load_and_process_image(
                prev_frame['main_image'], 
                size=(256, 256),
                justify=sequence['justify_main'],
                augment=sequence['augment'],
                aug_seed=sequence['aug_seed_main']
            ),
            load_and_process_image(
                current_frame['main_image'], 
                size=(256, 256),
                justify=sequence['justify_main'],
                augment=sequence['augment'],
                aug_seed=sequence['aug_seed_main']
            )
        ])
        
        wrist_images_obs = tf.stack([
            load_and_process_image(
                prev_frame['wrist_image'], 
                size=(128, 128),
                justify='center',
                augment=sequence['augment'],
                aug_seed=sequence['aug_seed_wrist']
            ),
            load_and_process_image(
                current_frame['wrist_image'], 
                size=(128, 128),
                justify='center',
                augment=sequence['augment'],
                aug_seed=sequence['aug_seed_wrist']
            )
        ])
        
        # Rest of the processing remains the same
        proprio_obs = tf.convert_to_tensor(sequence['proprioception'][:2], dtype=tf.float32)
        
        current_main = load_and_process_image(
            current_frame['main_image'], 
            size=(256, 256),
            justify=sequence['justify_main'],
            augment=sequence['augment'],
            aug_seed=sequence['aug_seed_main']
        )
        current_wrist = load_and_process_image(
            current_frame['wrist_image'], 
            size=(128, 128),
            justify='center',
            augment=sequence['augment'],
            aug_seed=sequence['aug_seed_wrist']
        )
        
        timestep_mask = tf.ones([2], dtype=tf.bool)
        action_pad_mask = tf.ones((2, 4, 7), dtype=tf.bool)
        
        actions = tf.convert_to_tensor(sequence['actions'], dtype=tf.float32)
        actions = tf.stack([actions, actions])
        
        return {
            'observation': {
                'image_primary': main_images_obs,
                'image_wrist': wrist_images_obs,
                'proprio': proprio_obs,
                'pad_mask_dict': {
                    'image_primary': timestep_mask,
                    'image_wrist': timestep_mask,
                    'proprio': timestep_mask
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
                'proprio': tf.TensorSpec(shape=(2, 7), dtype=tf.float32),
                'pad_mask_dict': {
                    'image_primary': tf.TensorSpec(shape=(2,), dtype=tf.bool),
                    'image_wrist': tf.TensorSpec(shape=(2,), dtype=tf.bool),
                    'proprio': tf.TensorSpec(shape=(2,), dtype=tf.bool)
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
        # "task_preprocessor.*", 
        # "observation_preprocessor.*"
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
    
    # Add proprioception tokenizer to model config
    model.config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    
    # Initialize action processor
    action_processor = ActionProcessor()
    
    # Load raw sequences first to compute statistics
    raw_sequences = []
    session_dirs = glob.glob(os.path.join(FLAGS.data_dir, "session_*"))
    logging.info(f"Found {len(session_dirs)} session directories")
    
    for session_dir in session_dirs:
        sequences = load_raw_session_data(session_dir)
        raw_sequences.extend(sequences)
        # logging.info(f"Loaded {len(sequences)} sequences from {os.path.basename(session_dir)}")
    
    logging.info(f"Total raw sequences loaded: {len(raw_sequences)}")

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
        
    logging.info(f"Total normalized sequences: {len(all_sequences)}")
    
    # Calculate approximate size of training dataset
    batches_per_epoch = len(all_sequences) // FLAGS.batch_size
    logging.info(f"With batch size {FLAGS.batch_size}, will have ~{batches_per_epoch} batches per epoch")
    
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
        return new_state, info, loss
    
    # Initialize running average
    ema_loss = None
    ema_alpha = 0.98
    
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
            pbar.update(1)
            
            # Checkpointing
            if (i + 1) % 1000 == 0:
                train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)
                gc.collect()
                jax.clear_caches()

if __name__ == "__main__":
    flags.FLAGS([""])  # Initialize flags
    app.run(main)