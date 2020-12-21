from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensor2tensor.data_generators import (
    problem,
    text_problems,
    text_encoder
)
from tensor2tensor.utils import metrics, misc_utils
from tensor2tensor.utils import registry
import tensorflow as tf
import os, re, getpass

RAW_DATA_PATH = f"./SCAN/"

REGISTERED_PROBLEMS = []


def scan_translate(commands):

    if 'and' in commands:
        x1 = commands.split("and")[0].strip()
        x2 = commands.split("and")[1].strip()
        return scan_translate(x1) + scan_translate(x2)
    if 'after' in commands:
        x1 = commands.split("after")[0].strip()
        x2 = commands.split("after")[1].strip()
        return scan_translate(x2) + scan_translate(x1)
    if 'twice' in commands:
        x = commands.split("twice")[0].strip()
        return scan_translate(x) + scan_translate(x)
    if 'thrice' in commands:
        x = commands.split("thrice")[0].strip()
        return scan_translate(x) + scan_translate(x) + scan_translate(x)
    if 'around left' in commands:
        x = commands.split("around left")[0].strip()
        if x == 'turn':
            return "".join(['LTURN ' for _ in range(4)])
        else:
            return "".join(['LTURN ' + scan_translate(x) for _ in range(4)])
    if 'around right' in commands:
        x = commands.split("around right")[0].strip()
        if x == 'turn':
            return "".join(['RTURN ' for _ in range(4)])
        else:
            return "".join(['RTURN ' + scan_translate(x) for _ in range(4)])
    if 'opposite left' in commands:
        x = commands.split("opposite left")[0].strip()
        if x == 'turn':
            return "".join(['LTURN ' for _ in range(2)])
        else:
            return "LTURN LTURN " + scan_translate(x)
    if 'opposite right' in commands:
        x = commands.split("opposite right")[0].strip()
        if x == 'turn':
            return "".join(['RTURN ' for _ in range(2)])
        else:
            return "RTURN RTURN " + scan_translate(x)
    if 'left' in commands and 'turn' not in commands:
        return "LTURN " + scan_translate(commands.split("left")[0].strip())
    if 'right' in commands and 'turn' not in commands:
        return "RTURN " + scan_translate(commands.split("right")[0].strip())
    if 'walk'==commands:
        return "WALK "
    if 'look'==commands:
        return "LOOK "
    if 'run'==commands:
        return "RUN "
    if 'jump'==commands:
        return "JUMP "
    if 'turn left'==commands:
        return "LTURN "
    if 'turn right'==commands:
        return "RTURN "


def generate_NLA_ops_action(commands):
    if 'and' in commands:
        x1 = commands.split("and")[0].strip()
        x2 = commands.split("and")[1].strip()
        return generate_NLA_ops_action(x1) + generate_NLA_ops_action(x2)
    if 'after' in commands:
        x1 = commands.split("after")[0].strip()
        x2 = commands.split("after")[1].strip()
        return generate_NLA_ops_action(x2) + generate_NLA_ops_action(x1)
    if 'twice' in commands:
        x = commands.split("twice")[0].strip()
        return generate_NLA_ops_action(x) + generate_NLA_ops_action(x)
    if 'thrice' in commands:
        x = commands.split("thrice")[0].strip()
        return generate_NLA_ops_action(x) + generate_NLA_ops_action(x) + generate_NLA_ops_action(x)
    else:
        return [1] + [0 for _ in scan_translate(commands.strip()).strip().split(" ")][1:]


def generate_NLA_ops_command(command):
    list_action = []
    command_seq = command.split(" ")
    command_len = len(command_seq)
    for i in range(command_len):
        if command_seq[i] in ["and", "after"]:
            list_action.append(1)
        elif i > 1 and command_seq[i-1] in ["and", "after"]:
            list_action.append(1)
        elif command_seq[i] in ["twice", "thrice"]:
            list_action.append(1)
        elif i > 1 and command_seq[i-1] in ["twice", "thrice"]:
            list_action.append(1)
        else:
            list_action.append(0)
    return list_action + [1]


class AlgorithmicSCAN(text_problems.Text2TextProblem):
    @property
    def vocab_filename(self):
        return "vocab.algorithmic_scan.32.tokens"

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def approx_vocab_size(self):
        return 2**5

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC,
            metrics.Metrics.ACC_PER_SEQ,
        ]

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    def target_prerpocess(self, target):
        target = re.sub("I_", "", target)
        target = re.sub("TURN_RIGHT", "RTURN", target)
        target = re.sub("TURN_LEFT", "LTURN", target)
        return target

    def read_SCAN_dataset(self, txt):
        with open(txt, 'r') as f:
            raw_data = f.readlines()
        instances = []
        for datum in raw_data:
            a, target = datum.split("OUT:")
            target = target[1:].strip()
            instances.append([a[4:].strip(), self.target_prerpocess(target)])
        return instances

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            txt_file = self.train_txt
        else:
            txt_file = self.test_txt

        instances = self.read_SCAN_dataset(txt_file)

        for example in instances:
            NLA_c = generate_NLA_ops_command(example[0])
            NLA_a = generate_NLA_ops_action(example[0]) + [1]  # for EOS
            yield {"inputs": example[0], "targets": example[1], "NLA_ops_command": NLA_c,
                   "NLA_ops_action": NLA_a}

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        return {
            "inputs": encoder,
            "NLA_ops_command": text_encoder.RealEncoder(),
            "NLA_ops_action": text_encoder.RealEncoder(),
            "targets": encoder
        }

    def example_reading_spec(self):
        data_fields = {"targets": tf.VarLenFeature(tf.int64)}
        data_fields.update({"NLA_ops_command": tf.VarLenFeature(tf.int64)})
        data_fields.update({"NLA_ops_action": tf.VarLenFeature(tf.int64)})
        if self.has_inputs:
            data_fields["inputs"] = tf.VarLenFeature(tf.int64)
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def preprocess_example(self, example, mode, hparams):
        example["NLA_ops_command"] = tf.one_hot(example["NLA_ops_command"], depth=3, dtype=tf.float32)
        example["NLA_ops_action"] = tf.one_hot(example["NLA_ops_action"], depth=3, dtype=tf.float32)
        return example


class AlgorithmicSCANSep(AlgorithmicSCAN):
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            txt_file = self.train_txt
        else:
            txt_file = self.test_txt

        instances = self.read_SCAN_dataset(txt_file)

        for example in instances:
            NLA_c = generate_NLA_ops_command(example[0])
            NLA_a = generate_NLA_ops_action(example[0])
            movements = []
            for action, movement in zip(NLA_a, example[1].split(" ")):
                if action == 1:
                    movements.append("<sep>")
                movements.append(movement)
            movements = " ".join(movements)
            yield {"inputs": example[0], "targets": movements, "NLA_ops_command": NLA_c,
                   "NLA_ops_action": NLA_a}


def _problems_to_register():
    all_problems = {}
    split_types = [i for i in os.listdir(RAW_DATA_PATH) if "split" in i]
    for split_type in split_types:
        cur_path = os.path.join(RAW_DATA_PATH, split_type)
        raw_data_txts = [i for i in os.listdir(cur_path) if "txt" in i]
        problems = ["_".join(j.split(".")[0].split("_")[2:]) for j in
                    list(set([i for i in raw_data_txts if "train" in i]))]
        for problem in problems:
            train_txt = os.path.join(cur_path, [i for i in raw_data_txts if "train" in i and problem in i][0])
            test_txt = os.path.join(cur_path, [i for i in raw_data_txts if "test" in i and problem in i][0])
            all_problems.update({problem: [train_txt, test_txt]})
    return all_problems


def _register_scan_problems():
    classes = [
        AlgorithmicSCAN,
        AlgorithmicSCANSep,
    ]
    for problem_name, txts in six.iteritems(_problems_to_register()):
        for class_ in classes:
            base_problem_class_name = misc_utils.camelcase_to_snakecase(class_.__name__)
            problem_class = type(f"{base_problem_class_name}_{problem_name}",
                                 (class_,), {
                                     "train_txt": txts[0],
                                     "test_txt": txts[1]
                                 })
            registry.register_problem(problem_class)
            REGISTERED_PROBLEMS.append(problem_class.name)

_register_scan_problems()



