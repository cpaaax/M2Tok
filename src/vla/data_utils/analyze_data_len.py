import os
import h5py
from tqdm import tqdm


HDF5_DIR = "/ailab/user/xuchunpu/code/collected_robotwin2.0_github_latest_data_filter_redundant_piper_12_tasks_randomized_clean_table"
tasks = [
    "beat_block_hammer",
    "click_bell",
    "handover_mic",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "pick_diverse_bottles",
    "place_mouse_pad",
    "place_container_plate",
    "place_phone_stand",
    "place_burger_fries",
    "shake_bottle",
    ]


file_paths = []
# sub_folders = [os.path.join(HDF5_DIR, f) for f in os.listdir(HDF5_DIR)]
for task in tasks:
    sub_folder = os.path.join(HDF5_DIR, task)
    sub_folder_files = sorted(
        [os.path.join(HDF5_DIR, sub_folder, file) for file in os.listdir(sub_folder) if file.endswith(".hdf5")])
    training_files = sub_folder_files[:1600]

    lens = []
    for file_path in tqdm(training_files):
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]  # an array that holds all the joint position values
            num_steps = qpos.shape[0]
            lens.append(num_steps)
    print(f"there are {len(lens)} files, the mean len for {sub_folder} is {sum(lens)/len(lens)}")


# beat_block_hammer is 103
# place_container_plate is 139
# place_dual_shoes is 202
# place_empty_cup is 148
# place_phone_stand is 110
# place_shoe is 135
# stack_blocks_two is 274
# handover_block is 245
# move_pillbottle_pad is 129
# pick_diverse_bottles is 111
# scan_objec is 155
# shake_bottle is 229
# adjust_bottle is 131
# move_stapler_pad is 135
# click_alarmclock is 76