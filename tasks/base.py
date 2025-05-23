import json
import logging
import random
import re
from collections import defaultdict

import torch
import numpy as np
import datasets

from datasets import Dataset

# from style_transfer_localLLM import generate_transfer_style
from anchor import hf_datasets_root
from tasks.loader import TokenizedForStyleRightPad
from utils.rng_ctx import RandomContext, EmptyContext
from utils.pca import PCA

from utils.forward_tracer import ForwardTrace
from utils.forward_tracer import ForwardTracer

logger = logging.getLogger("task")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BaseProbInference:
    def __init__(self, prompt_version):
        if prompt_version == "default":
            self.prompt_version = self.default_prompt_version()
        else:
            self.prompt_version = prompt_version

        self.raw_data_sample = None
        self.raw_data_dev = None

        self.can_be_stratified = False
        self.num_base_shot = 1

        self._rng_context = EmptyContext()

        self._cached_prefix = None
        self._cached_ex_list = None
        self._cahced_selected_exemplar = None
        self.shuffled_mapping = None

    def default_prompt_version(self):
        raise NotImplementedError

    def set_seed(self, seed):
        self._rng_context = RandomContext(seed=seed)

    def dataset_signature(self):
        raise NotImplementedError

    def dataset_part(self, part):
        return self.dataset_signature()[part]

    def dataset_preprocess(self, raw_data):
        raise NotImplementedError

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        raise NotImplementedError

    def paralell_style_promptify(self, query):
        raise NotImplementedError

    def shuffle_exemplars(self):
        prefix = self._cached_prefix
        ex_list = self._cached_ex_list

        ex_list_with_idx = list(enumerate(ex_list))
        with self._rng_context:
            random.shuffle(ex_list_with_idx)

        indices, ex_list = zip(*ex_list_with_idx)
        self.shuffled_mapping = indices

        return self.build_exemplar_from_examples(prefix, ex_list)

    def random_selected_exemplars(self, num_shots, seed, prefix = ""):

        with self._rng_context:
            num_shots = min(len(self.raw_data_sample), num_shots)

        random.seed(seed)
        indices = random.sample(range(len(self.raw_data_sample)), num_shots)

        sampled = [self.raw_data_sample[i] for i in indices]
        self._cahced_selected_exemplar = sampled

        ex_list = [e["query"] for e in sampled]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list.copy()
        return self.build_exemplar_from_examples(prefix, ex_list)

    def stratified_sampling(self, num_k_shots):
        num_shots = self.num_base_shot * num_k_shots

        if not self.can_be_stratified:
            logger.info("Cannot be stratified, fallback to random selection.")
            return self.random_selected_exemplars(num_shots)

        prefix = ""

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = num_shots // len(ans_set)
        residual = num_shots - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        selected_exemplar = [self.raw_data_sample[i] for i in selected_ids]
        self._cahced_selected_exemplar = selected_exemplar
        ex_list = [e["query"] for e in selected_exemplar]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def build_exemplar_from_examples(self, prefix, ex_list):
        s = prefix
        if len(s):
            s += self.exemplar_seperator()

        for query in ex_list:
            _, line = self.paralell_style_promptify(query)  # query, <query_with_answer>
            s += line + self.exemplar_seperator()
        return s

    def dataset_file_path(self, part):
        dataset_name, subset, split = self.dataset_part(part)
        dumped_folder = hf_datasets_root.joinpath("dumped")
        if not dumped_folder.exists():
            dumped_folder.mkdir(parents=True)

        if part == "sample":
            split = 'train' 
        if part == "result":
            split = 'test' 

        file_name = f"{dataset_name}-{subset}-{split}.jsonl"
        file_name = re.sub(r"[^\w_. -]", "_", file_name)
        return dumped_folder.joinpath(file_name)

    def do_load_part(self, part):
        f_path = self.dataset_file_path(part)
        print(f_path)
        if not f_path.exists():
            self.not_exist_download(part)
            return self.do_load_part(part)  # call once more
        else:
            with f_path.open("r") as f:
                raw_data = [json.loads(line) for line in f]
            data = self.dataset_preprocess(raw_data)
            logger.info(f"Data loaded: {part}.")
            return data

    def do_load(self):
        self.raw_data_sample = self.do_load_part("sample")
        self.raw_data_result = self.do_load_part("result")

    def not_exist_download(self, part):
        f_path = self.dataset_file_path(part)
        logger.info(f"{f_path} not exist, download from huggingface datasets hub...")

        dataset_name, subset, split = self.dataset_part(part)
        data = self.do_download(dataset_name, subset, split=split, cache_dir=str(hf_datasets_root))
        
        if part == "sample":
            data = data.train_test_split(test_size=0.4)['train']
        if part == "result":
            data = data.train_test_split(test_size=0.4)['test']

        data.to_json(f_path)
        logger.info(f"... success, saved at: {f_path}")

    @staticmethod
    def do_download(dataset_name, subset, split, cache_dir):
        if dataset_name == "shakespeare":
            train_original_file_path = "SHAKESPEARE_PATH"
            train_modern_file_path = "SHAKESPEARE_PATH"

            test_original_file_path = "SHAKESPEARE_PATH"
            test_modern_file_path = "SHAKESPEARE_PATH"

            with open(train_original_file_path, "r") as f:
                original_lines = f.readlines()
            with open(train_modern_file_path, "r") as f:
                modern_lines = f.readlines()

            with open(test_original_file_path, "r") as f:
                original_test_lines = f.readlines()
            with open(test_modern_file_path, "r") as f:
                modern_test_lines = f.readlines()
            
            if split == "train":
                raw_data = []
                for original, modern in zip(original_lines, modern_lines):
                    raw_data.append({"modern": modern, "original": original})
                logger.info("SHAKESPEARE loaded successfully.")
            elif split == "test":
                raw_data = []
                for original, modern in zip(original_test_lines, modern_test_lines):
                    raw_data.append({"modern": modern, "original": original})
                logger.info("SHAKESPEARE loaded successfully.")
            final_data = Dataset.from_dict({key: [d[key] for d in raw_data] for key in raw_data[0]})

            return final_data
        elif dataset_name == "gyafc_music":
            train_informal_file_path = "GYAFC_PATH"
            train_formal_file_path = "GYAFC_PATH"

            test_informal_file_path = "GYAFC_PATH"
            test_formal_file_path = "GYAFC_PATH"

            with open(train_informal_file_path, "r") as f:
                informal_lines = f.readlines()
            with open(train_formal_file_path, "r") as f:
                formal_lines = f.readlines()

            with open(test_informal_file_path, "r") as f:
                informal_test_lines = f.readlines()
            with open(test_formal_file_path, "r") as f:
                formal_test_lines = f.readlines()

            if split == "train":
                raw_data = []
                for informal, formal in zip(informal_lines, formal_lines):
                    raw_data.append({"informal": informal, "formal": formal})
                logger.info("GYAFC loaded successfully.")
            elif split == "test":
                raw_data = []
                for informal, formal in zip(informal_test_lines, formal_test_lines):
                    raw_data.append({"informal": informal, "formal": formal})
                logger.info("GYAFC loaded successfully.")
            final_data = Dataset.from_dict({key: [d[key] for d in raw_data] for key in raw_data[0]})
            return final_data
        
        elif dataset_name == "gyafc_family":
            train_informal_file_path = "GYAFC_PATH"
            train_formal_file_path = "GYAFC_PATH"

            test_informal_file_path = "GYAFC_PATH"
            test_formal_file_path = "GYAFC_PATH"

            with open(train_informal_file_path, "r") as f:
                informal_lines = f.readlines()
            with open(train_formal_file_path, "r") as f:
                formal_lines = f.readlines()

            with open(test_informal_file_path, "r") as f:
                informal_test_lines = f.readlines()
            with open(test_formal_file_path, "r") as f:
                formal_test_lines = f.readlines()

            if split == "train":
                raw_data = []
                for informal, formal in zip(informal_lines, formal_lines):
                    raw_data.append({"informal": informal, "formal": formal})
                logger.info("GYAFC loaded successfully.")
            elif split == "test":
                raw_data = []
                for informal, formal in zip(informal_test_lines, formal_test_lines):
                    raw_data.append({"informal": informal, "formal": formal})
                logger.info("GYAFC loaded successfully.")
            final_data = Dataset.from_dict({key: [d[key] for d in raw_data] for key in raw_data[0]})
            return final_data


        else:
            raw_data = datasets.load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
            logger.info("Download success.")
            return raw_data

    def mk_result_dataset(self, tokenizer, test=False, no_padding=False, prefix=''):
        set_seed(0)
        if len(self.raw_data_result) > 700:
            print("Randomly select 700 samples for test set.")
            self.raw_data_result = random.sample(self.raw_data_result, 700)
        else:
            self.raw_data_result = self.raw_data_result
        return TokenizedForStyleRightPad(self.raw_data_result, tokenizer, self.paralell_style_promptify, no_padding=no_padding, prefix=prefix), self.raw_data_result

    def mk_test_dataset(self, tokenzier):
        return self.mk_result_dataset(tokenzier)

    def split_train_test(self, dataset, dataset_name):
        if dataset_name == "paradetox":
            # randomly take 670 samples from dataset for test set, the rest is for training
            # randomly select 670 indices 
            eval_indices = random.sample(range(len(dataset)), 670)
            eval_set = [dataset[i] for i in eval_indices]
            train_indices = set(range(len(dataset))) - set(eval_indices)
            train_set = [dataset[i] for i in train_indices]
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        return train_set, eval_set

    def mk_dataset(self, dataset, tokenizer, no_padding=False, prefix=''):
        return TokenizedForStyleRightPad(dataset, tokenizer, self.paralell_style_promptify, no_padding=no_padding, prefix=prefix)

    def mk_sample_dataset(self, tokenizer, n_memory_samples, seed, no_padding=False, prefix=''):
        if n_memory_samples is None:
            raise ValueError("n_memory_samples should not be None.")
            dataset = TokenizedForStyleRightPad(
                self.raw_data_sample, tokenizer, self.paralell_style_promptify, no_padding=no_padding, prefix=prefix
            )
            return dataset, None  # Return None for indices when n_memory_samples is None
        else:
            # Generate random indices
            random.seed(seed)
            indices = random.sample(range(len(self.raw_data_sample)), n_memory_samples)
            # Select samples using the indices
            samples = [self.raw_data_sample[i] for i in indices]
            self._cahced_selected_exemplar = samples
            dataset = TokenizedForStyleRightPad(samples, tokenizer, self.paralell_style_promptify, no_padding=no_padding, prefix=prefix)
            self._cached_prefix = prefix
            self._cached_ex_list = samples.copy()

            return dataset, samples  # Return both the dataset and indices

    def mk_dev_dataset(self, tokenizer):
        sample_size = len(self.raw_data_result)

        ans_set = set(e["answer_idx"] for e in self.raw_data_sample)
        ans_map = defaultdict(list)
        for idx, e in enumerate(self.raw_data_sample):
            label = e["answer_idx"]
            ans_map[label].append(idx)

        per_label = sample_size // len(ans_set)
        residual = sample_size - per_label * len(ans_set)

        selected_ids = []
        with self._rng_context:
            for label, all_ids in ans_map.items():
                selected = random.sample(all_ids, per_label)
                selected_ids.extend(selected)

            remain_ids = set(range(len(self.raw_data_sample))) - set(selected_ids)
            residual_selected = random.sample(remain_ids, residual)
            selected_ids.extend(residual_selected)
            random.shuffle(selected_ids)

        self.raw_data_dev = [self.raw_data_sample[i] for i in selected_ids]
        return TokenizedForStyleRightPad(self.raw_data_dev, tokenizer, self.paralell_style_promptify)

    def mk_finetune_dataset(self, tokenizer, mode = 'ft'):
        selected_exemplar = self._cahced_selected_exemplar
        assert (selected_exemplar != None), "No demonstration is selected yet, run stratified_sampling first! \n"
        return TokenizedForStyleRightPad(selected_exemplar, tokenizer, self.paralell_style_promptify, mode=mode)

    def mk_result_dataset_with_demostration(self, tokenizer, exemplar_str, no_padding=False):
        def add_demostration(query, return_reference = False, Instruction = ''):
            if return_reference:
                with_query, with_query_and_paraphrase, references = self.paralell_style_promptify(query, return_reference=return_reference, Instruction=Instruction)
                with_query = with_query.replace(Instruction,"")
                with_query_and_paraphrase = with_query_and_paraphrase.replace(Instruction,"")
                return f"{exemplar_str}{with_query}", f"{exemplar_str}{with_query_and_paraphrase}", references
            else:
                with_query, with_query_and_paraphrase = self.paralell_style_promptify(query, return_reference=return_reference, Instruction=Instruction)
                with_query = with_query.replace(Instruction,"")
                with_query_and_paraphrase = with_query_and_paraphrase.replace(Instruction,"")
                return f"{exemplar_str}{with_query}", f"{exemplar_str}{with_query_and_paraphrase}"

        return TokenizedForStyleRightPad(self.raw_data_result, tokenizer, add_demostration, no_padding=no_padding)
    
    @staticmethod
    def standardize(tensor, dim=0):
        means = tensor.mean(dim=dim, keepdim=True)
        stds = tensor.std(dim=dim, unbiased=False, keepdim=True)
        return (tensor - means) / stds

    @staticmethod
    def get_hiddenstates(model, inputs):
        h_all = []

        for example_id in range(len(inputs)):
            embeddings_for_all_styles= []
            for style_id in range(len(inputs[example_id])):
                forward_trace = ForwardTrace()
                context_manager = ForwardTracer(model, forward_trace)
                with torch.no_grad(), context_manager:
                    _ = model(
                    input_ids=torch.tensor(inputs[example_id][style_id]['input_ids']).unsqueeze(0).cuda(), 
                    attention_mask = torch.tensor(inputs[example_id][style_id]['attention_mask']).unsqueeze(0).cuda(), 
                    output_attentions=False,
                    output_hidden_states=False
                    )
                    h = forward_trace.residual_stream.hidden
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,-1])
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        return h_all

    
    @staticmethod
    def obtain_icv(model, inputs, rank=1):
        hidden_states = BaseProbInference.get_hiddenstates(model, inputs) #each element, layer x len_tokens x dim
        
        num_demonstration = len(hidden_states)
        neg_all = []
        pos_all = []

        hidden_states_all = []

        # getting the dataset of h(y) - h(x), yet to be decide which is the ICV
        for demonstration_id in range(num_demonstration):
            h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
            hidden_states_all.append(h)
            neg_all.append(hidden_states[demonstration_id][0].view(-1))
            pos_all.append(hidden_states[demonstration_id][1].view(-1))
        fit_data = torch.stack(hidden_states_all)
        neg_emb = torch.stack(neg_all).mean(0)
        pos_emb = torch.stack(pos_all).mean(0)

        # test_emb = pos_emb - neg_emb
        # test_emb = test_emb.view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))

        # print(test_emb.shape)

        pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
        eval_data =  pca.transform(fit_data.float())
        h_pca = pca.inverse_transform(eval_data) 
        direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
        
        return direction, (neg_emb).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    

def obtain_memory_icv(model, inputs, rank=1):
    hidden_states = BaseProbInference.get_hiddenstates(model, inputs) #each element, layer x len_tokens x dim

    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []

    hidden_states_all = []

    # getting the dataset of h(y) - h(x), yet to be decide which is the ICV
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    neg_emb = torch.stack(neg_all).mean(0)
    pos_emb = torch.stack(pos_all).mean(0)

    direction = pos_emb - neg_emb
    direction = direction.view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))

    return direction
