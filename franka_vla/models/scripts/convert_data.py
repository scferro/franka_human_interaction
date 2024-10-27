import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

# ====== EDIT THESE PATHS ======
INPUT_DIR = "/home/scferro/Documents/final_project/training_data_vla"  # Directory containing session_* folders
OUTPUT_FILE = "/home/scferro/Documents/final_project/training_data_vla/dataset.tfrecord"  # Where to save the TFRecord file
# ============================

def load_and_process_session(session_path):
    """Load and process a single session directory."""
    # Load the jsonl file
    data_path = os.path.join(session_path, "data.jsonl")
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    # Process each frame
    session_data = []
    for line in lines:
        frame = json.loads(line)
        
        # Read images
        main_img_path = os.path.join(session_path, frame['main_image'])
        wrist_img_path = os.path.join(session_path, frame['wrist_image'])
        
        main_img = Image.open(main_img_path)
        wrist_img = Image.open(wrist_img_path)
        
        # Convert images to arrays and normalize
        main_array = np.array(main_img) / 255.0
        wrist_array = np.array(wrist_img) / 255.0
        
        # Create example
        example = {
            'main_image': main_array,
            'wrist_image': wrist_array,
            'action': np.array(frame['action'], dtype=np.float32),
            'prompt': frame['prompt']
        }
        session_data.append(example)
    
    return session_data

def create_action_chunks(session_data, chunk_size=4):
    """Create overlapping chunks of actions."""
    chunks = []
    for i in range(0, len(session_data) - chunk_size + 1):
        chunk = session_data[i:i + chunk_size]
        
        # Stack the actions
        actions = np.stack([frame['action'] for frame in chunk])
        
        # Use the first frame's images and prompt
        example = {
            'main_image': chunk[0]['main_image'],
            'wrist_image': chunk[0]['wrist_image'],
            'action': actions,
            'prompt': chunk[0]['prompt']
        }
        chunks.append(example)
    
    return chunks

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tf_example(example):
    """Convert a single example to tf.Example format."""
    # Serialize numpy arrays
    main_image_raw = tf.io.serialize_tensor(example['main_image'])
    wrist_image_raw = tf.io.serialize_tensor(example['wrist_image'])
    action_raw = tf.io.serialize_tensor(example['action'])
    
    feature = {
        'main_image': _bytes_feature(main_image_raw),
        'wrist_image': _bytes_feature(wrist_image_raw),
        'action': _bytes_feature(action_raw),
        'prompt': _bytes_feature(example['prompt'].encode()),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
    # Verify input directory exists
    if not os.path.exists(INPUT_DIR):
        raise ValueError(f"Input directory {INPUT_DIR} does not exist")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all session directories
    session_dirs = glob.glob(os.path.join(INPUT_DIR, "session_*"))
    
    if not session_dirs:
        raise ValueError(f"No session directories found in {INPUT_DIR}")
    
    print(f"Found {len(session_dirs)} session directories")
    
    # Process sessions and write to TFRecord
    with tf.io.TFRecordWriter(OUTPUT_FILE) as writer:
        for session_dir in tqdm(session_dirs):
            try:
                # Load and process session data
                session_data = load_and_process_session(session_dir)
                
                # Create action chunks
                chunks = create_action_chunks(session_data, chunk_size=4)
                
                # Write examples
                for chunk in chunks:
                    tf_example = create_tf_example(chunk)
                    writer.write(tf_example.SerializeToString())
            except Exception as e:
                print(f"Error processing {session_dir}: {str(e)}")

if __name__ == "__main__":
    main()