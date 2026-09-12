import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np
import imageio
from PIL import Image
import random

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self, data_input_path, action_chunk_size=16, img_history_size=1, state_dim=14,
                 batch_transform=None,
                 instruction_path=None) -> None:

        HDF5_DIR = data_input_path
        # HDF5_DIR = model_config["data_path"]
        self.DATASET_NAME = "agilex"
        self.batch_transform = batch_transform
        self.file_paths = []



        sub_folders = [os.path.join(HDF5_DIR, folder) for folder in os.listdir(HDF5_DIR)]

        self.data_all = []

        self.traj_lens_all = []   # [0,0,0,1,1,1, ...], where values represent trajectory indices, and the count of each index equals the average length of this task.
        traj_idx = 0  # Tracks trajectory indices

        print("calculate trajectory length for each task")
        for sub_folder in sub_folders:
            sub_folder_files = sorted(
                [os.path.join(HDF5_DIR, sub_folder, file) for file in os.listdir(sub_folder) if file.endswith(".hdf5")])
            # Calculate the average trajectory length for each task
            sub_folder_traj_len = []
            for file in sub_folder_files:
                with h5py.File(file, 'r') as f:
                    traj_len = f['observations']['qpos'].shape[0]
                sub_folder_traj_len.append(traj_len)
            cur_folder_traj_avg_len = sum(sub_folder_traj_len)//len(sub_folder_traj_len)


            for file_path in sub_folder_files:
                cur_path_data = [file_path]
                self.data_all.append(cur_path_data)
                self.traj_lens_all.extend([traj_idx] * cur_folder_traj_avg_len)
                traj_idx+=1


        self.CHUNK_SIZE = action_chunk_size
        self.IMG_HISORY_SIZE = img_history_size
        self.STATE_DIM = state_dim
        self.step_size_per_sample = self.CHUNK_SIZE * self.IMG_HISORY_SIZE
        self.instruction_path = instruction_path


    def __len__(self):
        return len(self.traj_lens_all)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def __getitem__(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                # file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
                # file_path = np.random.choice(self.file_paths)
                cur_data = np.random.choice(self.data_all)
            else:
                traj_idx = self.traj_lens_all[index]
                cur_data = self.data_all[traj_idx]

            valid, sample = self.parse_hdf5_file(cur_data)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.data_all))

    def parse_hdf5_file(self, cur_data):
        file_path = cur_data[0]
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 60:
                return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # We randomly sample an img_history_size timestep

            step_id = np.random.randint(first_idx - 1,
                                        num_steps - self.step_size_per_sample)
            # Load the instruction
            instructions_path = os.path.join(self.instruction_path, file_path.split('/')[-2],
                                             # file_path.split('/')[-2] is task
                                             file_path.split('/')[-1].replace(".hdf5", ".json"))

            instructions = json.load(open(instructions_path, "r"))["seen"]
            instruction = random.choice(instructions)

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            target_qpos = f['action'][step_id:step_id + self.step_size_per_sample]

            # Parse the state and action
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
            delta_actions = target_qpos - target_qpos[:1]  # delta joint actions

            actions = delta_actions

            # Parse the images
            def parse_img(key, pred_future_img=False):
                imgs = []
                if pred_future_img:
                    for i in range(step_id, step_id + self.step_size_per_sample+1, self.CHUNK_SIZE):
                        img = f['observations']['images'][key][i]
                        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                        imgs.append(img)
                else:
                    img = f['observations']['images'][key][step_id]
                    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgs.append(img)
                return imgs

            cam_high = parse_img('cam_high', pred_future_img=False)

            sample = self.batch_transform(meta, actions, cam_high, state)

            return True, sample


    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]  # an array that holds all the joint position values
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )
            target_qpos = f['action'][:] / np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )

            # Parse the state and action
            state = qpos[first_idx - 1:]
            action = target_qpos[first_idx - 1:]



            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
