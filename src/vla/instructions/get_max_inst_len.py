import os
import json
import tiktoken

from transformers import AutoTokenizer
from tqdm import tqdm
# 加载 Qwen2.5 的 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/chunpu/vla/VLA_weights_/Qwen2.5-0.5B")



# Token 计数
input_path = "/mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions_randomized"

folders =  [os.path.join(input_path, folder) for folder in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, folder))]

for folder in folders:
    files = os.listdir(folder)

    lens = []
    lens_ = []

    for file in tqdm(files):
        # print(file)
        data = json.load(open(os.path.join(folder, file), "r"))["seen"]

        max_len = max([len(tokenizer.tokenize(item)) for item in data])
        max_len_ = max([len(item.split(" ")) for item in data])

        lens.append(max_len)
        lens_.append(max_len_)
    print(f" {folder} max len is tokenizer {max(lens)}, split is {max(lens_)} ")



# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:13<00:00, 136.44it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/shake_bottle max len is tokenizer 20, split is 18
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:15<00:00, 118.28it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/move_can_pot max len is tokenizer 33, split is 26
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1799/1799 [00:14<00:00, 123.80it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/place_mouse_pad max len is tokenizer 26, split is 24
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:13<00:00, 131.45it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/move_pillbottle_pad max len is tokenizer 22, split is 20
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:13<00:00, 136.77it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/place_phone_stand max len is tokenizer 19, split is 15
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:15<00:00, 117.56it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/pick_diverse_bottles max len is tokenizer 29, split is 25
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:13<00:00, 137.53it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/move_playingcard_away max len is tokenizer 19, split is 18
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:14<00:00, 124.85it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/handover_mic max len is tokenizer 23, split is 21
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:15<00:00, 114.45it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/place_container_plate max len is tokenizer 26, split is 23
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:17<00:00, 102.25it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/place_burger_fries max len is tokenizer 34, split is 32
#  62%|████████████████████████████████████████████████████████████████████████████████████████████▎                                                        | 1115/1800 [00:08<00:04, 140.15it/s]
#  67%|███████████████████████████████████████████████████████████████████████████████████████████████████▋                                                 | 1204/1800 [00:08<00:04, 138.84it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:12<00:00, 138.74it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/click_bell max len is tokenizer 18, split is 17
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [00:12<00:00, 142.97it/s]
#  /mnt/workspace/chunpu/RoboTwin2.0_data_related_multi-task/instructions_related/piper_12_tasks_instructions/beat_block_hammer max len is tokenizer 18, split is 16