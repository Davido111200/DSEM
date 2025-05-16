import gymnasium
from gymnasium import spaces
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Text,
    Tuple,
)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import evaluate
import torch
import random
from itertools import combinations
from tqdm import tqdm
from torch.nn import functional as F

from models import build_model, build_tokenizer
from common import mk_parser
from tasks import load_task

from tasks.base import obtain_memory_icv

from utils.llm_layers import add_icv_layers, remove_icv_layers
from utils.forward_tracer import ForwardTrace
from utils.forward_tracer import ForwardTracer


from LIVE.inference import generate_answers, get_icv


from parlai.utils.safety import OffensiveLanguageClassifier


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def get_inverse_weight(chunk_index, offset=1):
    """
    Weight decays as 1/(index + offset), ensuring the first chunk is weighted highest.
    """
    return 1.0 / (chunk_index + offset)


class MemEnvTest(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, raw_train_data):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.raw_train_data = raw_train_data

        # build model and tokenizer
        model_to_hidden = {
            "llama-2": 4096,
            "falcon": 4544, 
            "llama-2-13b": 5120,
            "falcon-11b": 4096,
            "gemma-9b": 3584,
            "mistral-7b": 4096,
        }
        
        model_to_n_layers = {
            "llama-2": 32,
            "falcon": 32,
            "llama-2-13b": 40,
            "falcon-11b": 60,
            "gemma-9b": 42,
            "mistral-7b": 32,
        }

        self.n_layers = model_to_n_layers[self.model_name]
        self.hidden_size = model_to_hidden[self.model_name]

        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.hidden_size,),
            dtype=float
        )

        # state is discrete
        self.action_space = spaces.Discrete(1)

        self.current_data_idx = 0
        self.current_reward = 0
        self.steps = 0
        
    def _get_train_dataset(self):
        return self.train_dataset

    def _get_obs(self):
        return {"previous_actions": self.previous_actions, "current_state": self.current_state, "current_ts": self.steps}

    def _get_current_state(self):
        return self.current_state

    def _get_hidden_state(self, inputs):
        text = [inputs]  # assuming the text is in 'text' field
        encoded_inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            last_hidden_state = self.model(**encoded_inputs, output_hidden_states=True).hidden_states[-1]
        
        weights_for_non_padding = encoded_inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        return sentence_embeddings.cpu().detach().numpy()[0]



    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        original_sentence = self.raw_train_data[self.current_data_idx]['query'][0]

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(original_sentence)

        state = self._get_current_state()
        return state, {}

    def step(self, action):
        original_sentence, reference = self.raw_train_data[self.current_data_idx]['query'][0], self.raw_train_data[self.current_data_idx]['query'][1]

        # Get hidden states and ICV (keep existing logic)
        if self.args.model_type == "falcon":
            if self.args.dataset == "gyafc_music":
                clean_input = original_sentence.replace("\n","").replace("'","").replace('(',"").replace(')',"").replace('-',"").lower()
            else:
                clean_input = original_sentence.replace("\n","").replace("'","").lower()
        else:
            clean_input = original_sentence.replace("\n","")
        self.current_state = self._get_hidden_state(clean_input)
        
        icv = obtain_memory_icv(self.model, self.train_dataset.tokenize_each_demonstration(
            [(original_sentence, reference)], prefix=("", "")), rank=1
        )
        icv = icv[1:]
        add_icv_layers(self.model, torch.stack([icv], dim=1).cuda(), [self.args.lam])

        if 'llama-2' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [1642, 13492, 26036, 29908, self.tokenizer.encode('.10')[-1]]
            }
        elif 'falcon' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 10,
                'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166]
            }
        elif 'llama-2-13b' in self.args.model_type:
            gen_args = {
                        'temperature': 0.45,
                        'do_sample': True,
                        'top_k': 0,
                        'top_p': 1.0,
                        'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
                    }
        elif 'falcon-11b' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 5,
                'eos_token_id':[self.tokenizer.encode('\n')[-1], self.tokenizer.encode('<|endoftext|>')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1], self.tokenizer.encode('}')[-1]],
                }
        elif 'gemma-9b' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
            }
        elif 'mistral-7b' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
            }
        else:
            gen_args = {}

        # Modified generation with attention capture
        with torch.no_grad():
            input_id = self.train_dataset[self.current_data_idx][0].unsqueeze(0)
            attention_mask = self.train_dataset[self.current_data_idx][1].unsqueeze(0)
            
            # Generate with attention capture
            generation_output = self.model.generate(
                input_ids=input_id.cuda(),
                attention_mask=attention_mask.cuda(),
                max_new_tokens=32,
                pad_token_id=self.tokenizer.eos_token_id,
                output_attentions=True,  # Critical addition
                return_dict_in_generate=True,
                **gen_args,
            )

            # Extract attention patterns
            decoded_output = self.tokenizer.decode(generation_output.sequences[0][len(input_id[0]):])
            
            # Clean output (keep existing processing)
            clean_output = decoded_output.replace("\n","").replace("{","").replace("}","").replace('"','').strip('".')\
                            .replace(',,','').replace('original','').replace('Original','')\
                            .split('rewritten')[0].split('revised')[0].replace('10','').split('.')[0]
            
            # Tokenize the clean output for chunking
            tokenized_output = self.tokenizer.tokenize(clean_output)

            # Function to chunk the
            # tokenized output into chunks of self.args.chunk_size tokens
            def chunk_tokens(tokens):
                return [tokens[0:i + self.args.chunk_size] for i in range(0, len(tokens), self.args.chunk_size)]
            
            # Create chunks of self.args.chunk_size tokens
            token_chunks = chunk_tokens(tokenized_output)

            # Add metadata context for retrieval
            chunks_with_metadata = {'sentence': original_sentence, 'steering_vector': icv.detach().clone(), 'chunks': {'chunk_sentence': [], "chunk_length": [], 'full_chunk_length': len(tokenized_output)}}
            for chunk in token_chunks:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                chunks_with_metadata['chunks']['chunk_sentence'].append(chunk_text)
                chunks_with_metadata['chunks']['chunk_length'].append(len(chunk))
        
        # Existing cleanup and return
        self.current_data_idx += 1
        terminated = (self.steps >= self.args.max_steps)
        remove_icv_layers(self.model)

        return self._get_current_state(), self.current_reward, terminated, chunks_with_metadata



class MemEnvEvalTest(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, raw_eval_data ,memory, global_icv):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.raw_eval_data = raw_eval_data
        self.memory = memory

        print(len(eval_dataset))
        print(len(raw_eval_data))

        # build model and tokenizer
        model_to_hidden = {
            "llama-2": 4096,
            "falcon": 4544, 
            "llama-2-13b": 5120,
            "falcon-11b": 4096,
            "gemma-9b": 3584,
            "mistral-7b": 4096,
        }
        
        model_to_n_layers = {
            "llama-2": 32,
            "falcon": 32,
            "llama-2-13b": 40,
            "falcon-11b": 60,
            "gemma-9b": 42,
            "mistral-7b": 32,
        }
        self.n_layers = model_to_n_layers[self.model_name]
        self.hidden_size = model_to_hidden[self.model_name]        

        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.n_layers, self.hidden_size),
            dtype=float
        )

        # state is discrete
        self.action_space = spaces.Discrete(1)

        self.global_icv = global_icv

        self.current_data_idx = 0
        self.current_reward = 0
        self.steps = 0

    def calculate_context_ratio(self, attentions, context_length):
        """
        Calculate the ratio of attention focused on the context tokens.
        
        Parameters:
            attentions (torch.Tensor): Attention scores, shape [batch_size, num_heads, seq_len, seq_len].
            context_length (int): Number of context tokens.

        Returns:
            torch.Tensor: Mean context ratio for all heads.
        """
        # Sum attention weights over context tokens
        context_attention = attentions[..., :context_length]  # Focus on context tokens
        context_sum = context_attention.sum(dim=-1)  # Sum over context tokens

        # Total attention per head
        total_attention = attentions.sum(dim=-1)

        # Avoid division by zero with epsilon
        epsilon = 1e-12
        context_ratio = context_sum / (total_attention + epsilon)

        # Average across all heads and batch
        avg_context_ratio = context_ratio.mean(dim=(0, 1))  # Average over heads and batch
        return avg_context_ratio


    def _get_train_dataset(self):
        return self.train_dataset

    def _get_current_state(self):
        return self.current_state

    def _get_hidden_state(self, inputs):
        text = [inputs]  # assuming the text is in 'text' field
        encoded_inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            last_hidden_state = self.model(**encoded_inputs, output_hidden_states=True).hidden_states[-1]
        
        weights_for_non_padding = encoded_inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        return sentence_embeddings.cpu().detach().numpy()[0]


    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(original_sentence)

        state = self._get_current_state()

        # update the current training sample
        return state, {}


    def step(self, action):
        original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]
        reference = self.raw_eval_data[self.current_data_idx]['query'][1]
        
        eval_data = self.eval_dataset[self.current_data_idx]
        self.current_state = self._get_hidden_state(original_sentence)

        current_sentence_with_ins = self.tokenizer.decode(eval_data[0], skip_special_tokens=True)

        if 'llama-2' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [1642, 13492, 26036, 29908,self.tokenizer.encode('.10')[-1], self.tokenizer.encode('\n')[-1]]
            }
        elif 'falcon' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 10,
                'eos_token_id': [104, 193, 1001, 25, 1702, 18858, 3166, self.tokenizer.encode('\n')[-1], self.tokenizer.encode('}')[-1]]
            }
        elif 'llama-2-13b' in self.args.model_type:
            gen_args = {
                        'temperature': 0.45,
                        'do_sample': True,
                        'top_k': 0,
                        'top_p': 1.0,
                        'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
                    }
        elif 'falcon-11b' in self.args.model_type:
            gen_args = {
                'do_sample': False,
                'num_beams': 5,
                'eos_token_id':[self.tokenizer.encode('\n')[-1], self.tokenizer.encode('<|endoftext|>')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1], self.tokenizer.encode('}')[-1]],
                }
        elif 'gemma-9b' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
            }
        elif 'mistral-7b' in self.args.model_type:
            gen_args = {
                'temperature': 0.45,
                'do_sample': True,
                'top_k': 0,
                'top_p': 1.0,
                'eos_token_id': [self.tokenizer.encode('\n')[-1], self.tokenizer.encode('".')[-1], self.tokenizer.encode('original')[-1], self.tokenizer.encode('Original')[-1], self.tokenizer.encode('"')[-1], self.tokenizer.encode('0')[-1]],
            }
        else:
            gen_args = {}
            
        cur_num_token = 0
        
        # init cur_answer_id as null
        cur_answer_id = torch.tensor([[]])
        cur_answer = ""
        while cur_num_token < 32:
            if self.args.chunk_vector:
                icv, uncertainty = self.memory.read_chunk(self.args, self._get_hidden_state(original_sentence), original_sentence, cur_answer, distance_metric=self.args.distance_metric, k_chunk=self.args.n_neighbor_chunks, k_sentence=self.args.n_neighbors)
            else:
                icv, uncertainty = self.memory.read(self.args, self._get_hidden_state(original_sentence), original_sentence, cur_answer, distance_metric=self.args.distance_metric, k_chunk=self.args.n_neighbor_chunks, k_sentence=self.args.n_neighbors)
            

            if self.args.mix_strat == "hard":
                if uncertainty < self.args.mix_hard_threshold:
                    add_icv_layers(self.model, torch.stack([icv],dim=1).cuda(), [self.args.lam])
                else:
                    add_icv_layers(self.model, torch.stack([self.global_icv],dim=1).cuda(), [self.args.lam])
            elif self.args.mix_strat == "soft":
                if self.args.tune_weight > 0:
                    soft_icv = (1 - self.args.tune_weight) * icv + self.args.tune_weight * self.global_icv
                elif self.args.only_memory is True:
                    soft_icv = icv
                else:
                    soft_icv = (1 - uncertainty) * icv + uncertainty * self.global_icv
                add_icv_layers(self.model, torch.stack([soft_icv],dim=1).cuda(), [self.args.lam])
            else:
                raise ValueError(f"Unknown mix strategy: {self.args.mix_strat}")
            
            with torch.no_grad():
                # Tokenize input
                input_id = self.tokenizer(current_sentence_with_ins, return_tensors='pt')['input_ids']
                attention_mask = self.tokenizer(current_sentence_with_ins, return_tensors='pt')['attention_mask']
                
                # Generate only self.args.chunk_size new token per iteration
                generation_output = self.model.generate(
                    input_ids=input_id.cuda(),
                    attention_mask=attention_mask.cuda(),
                    max_new_tokens=self.args.chunk_size,  # Generate self.args.chunk_size token at a time
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_args,
                )
                total_generated_tokens = generation_output.shape[1] - input_id.shape[1]
                next_tokens_id = generation_output[0, -min(self.args.chunk_size, total_generated_tokens):]

                # Stop if the next token is EOS
                if any(next_tokens_id) in gen_args['eos_token_id']:
                    remove_icv_layers(self.model)
                    # Append the next tokens to input_id
                    input_id = torch.cat([input_id, next_tokens_id.view(1, -1).cpu()], dim=1)
                    # Append the next tokens to current_answer_id
                    cur_answer_id = torch.cat([cur_answer_id, next_tokens_id.view(1, -1).cpu()], dim=1)


                    # get the next token
                    current_sentence_with_ins = self.tokenizer.decode(input_id[0], skip_special_tokens=True)
                    cur_answer = self.tokenizer.decode(cur_answer_id[0], skip_special_tokens=True)

                    break
                
                # Append the next tokens to input_id
                input_id = torch.cat([input_id, next_tokens_id.view(1, -1).cpu()], dim=1)
                # Append the next tokens to current_answer_id
                cur_answer_id = torch.cat([cur_answer_id, next_tokens_id.view(1, -1).cpu()], dim=1)


                # get the next token
                current_sentence_with_ins = self.tokenizer.decode(input_id[0], skip_special_tokens=True)
                if self.args.model_type in ["falcon", "falcon-11b", "gemma-9b", "mistral-7b"]:
                    cur_answer = self.tokenizer.decode(cur_answer_id[0].to(torch.int), skip_special_tokens=True)
                else:                    
                    cur_answer = self.tokenizer.decode(cur_answer_id[0], skip_special_tokens=True)
                
                # Increment the token count by the size of next_tokens_id
                cur_num_token += len(next_tokens_id)
            remove_icv_layers(self.model)



        # get the part to the right of Paraphrase, and get the text in between the first and last quotation marks
        generated_output = current_sentence_with_ins.split('Paraphrase')[1].split('"')[1]
        self.current_reward = None
        self.current_data_idx += 1

        return self._get_current_state(), self.current_reward, True, {"original_sentence": original_sentence, "icv": icv.detach().clone(), "generation": generated_output, "reference": reference}

    def get_vector(self, text, cur_answer):
        icv, uncertainty = self.memory.read(self.args, self._get_hidden_state(text), text, cur_answer, distance_metric=self.args.distance_metric, k_chunk=self.args.n_neighbor_chunks, k_sentence=self.args.n_neighbors)
        return icv, uncertainty
    



class MemEnvEvalTestArithmetic(gymnasium.Env):
    def __init__(self, args, model_name, model, tokenizer, train_dataset, eval_dataset, raw_eval_data ,memory, global_icv):
        self.args = args
        self.model_name = model_name    
        self.model = model
        self.tokenizer = tokenizer

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.raw_eval_data = raw_eval_data
        self.memory = memory

        print(len(eval_dataset))
        print(len(raw_eval_data))

        # build model and tokenizer
        model_to_hidden = {
            "llama-2": 4096,
            "falcon": 4544, 
            "llama-2-13b": 5120,
            "falcon-11b": 4096,
            "gemma-9b": 3584,
            "mistral-7b": 4096,
        }
        
        model_to_n_layers = {
            "llama-2": 32,
            "falcon": 32,
            "llama-2-13b": 40,
            "falcon-11b": 60,
            "gemma-9b": 42,
            "mistral-7b": 32,
        }
        self.n_layers = model_to_n_layers[self.model_name]
        self.hidden_size = model_to_hidden[self.model_name]        

        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.n_layers, self.hidden_size),
            dtype=float
        )

        self.raw_text = "What the hell is this?"
        self.cur_answer = ""

        # state is discrete
        self.action_space = spaces.Discrete(1)

        self.global_icv = global_icv

        self.current_data_idx = 0
        self.current_reward = 0
        self.steps = 0

    def calculate_context_ratio(self, attentions, context_length):
        """
        Calculate the ratio of attention focused on the context tokens.
        
        Parameters:
            attentions (torch.Tensor): Attention scores, shape [batch_size, num_heads, seq_len, seq_len].
            context_length (int): Number of context tokens.

        Returns:
            torch.Tensor: Mean context ratio for all heads.
        """
        # Sum attention weights over context tokens
        context_attention = attentions[..., :context_length]  # Focus on context tokens
        context_sum = context_attention.sum(dim=-1)  # Sum over context tokens

        # Total attention per head
        total_attention = attentions.sum(dim=-1)

        # Avoid division by zero with epsilon
        epsilon = 1e-12
        context_ratio = context_sum / (total_attention + epsilon)

        # Average across all heads and batch
        avg_context_ratio = context_ratio.mean(dim=(0, 1))  # Average over heads and batch
        return avg_context_ratio


    def _get_train_dataset(self):
        return self.train_dataset

    def _get_current_state(self):
        return self.current_state

    def _get_hidden_state(self, inputs):
        text = [inputs]  # assuming the text is in 'text' field
        encoded_inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        with torch.no_grad():
            last_hidden_state = self.model(**encoded_inputs, output_hidden_states=True).hidden_states[-1]
        
        weights_for_non_padding = encoded_inputs.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0)

        sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens

        return sentence_embeddings.cpu().detach().numpy()[0]


    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)

        # reset the total time steps
        self.steps = 0

        # reset the previous actions
        self.previous_actions = []

        # reset previous reward
        self.current_reward = 0

        original_sentence = self.raw_eval_data[self.current_data_idx]['query'][0]

        # get hidden state of the sentence
        self.current_state = self._get_hidden_state(original_sentence)

        state = self._get_current_state()

        # update the current training sample
        return state, {}


    def step(self, action):
        icv, uncertainty = self.memory.read(self.args, self._get_hidden_state(self.raw_text), self.raw_text, self.cur_answer, distance_metric=self.args.distance_metric, k_chunk=self.args.n_neighbor_chunks, k_sentence=self.args.n_neighbors)
        return None, 0, True, {"icv": icv.detach().clone(), "uncertainty": uncertainty, "text": self.raw_text, "answer": self.cur_answer}
    

