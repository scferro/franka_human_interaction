# Share the positions of the last block placed in each category
geometry_msgs/Pose last_block_pose              # The position and orientation of the last placed block
int32 last_block_category                           # The category of the last placed block (integer from 0 to 3)
geometry_msgs/Pose[] next_positions             # Array of poses for the next block in each category
int32 last_block_index                           # The category of the last placed block (integer from 0 to 3)
