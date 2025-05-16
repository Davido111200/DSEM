import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import math
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_bm25(text):
    # Basic tokenization: lowercase and split by spaces (you can improve this with regex or other libraries)
    return text.lower().split()

def get_bm25_scores(query, docs):
    query_tokens = tokenize_bm25(query)
    doc_tokens = [tokenize_bm25(doc) for doc in docs]
    bm25 = BM25Okapi(doc_tokens)
    return bm25.get_scores(query_tokens)


def rank_rrf(k, rank1, rank2):
    """
    Compute Reciprocal Rank Fusion (RRF) for multiple ranking lists.

    Parameters:
    - k (int): Smoothing factor (default: 60)

    Returns:
    - List of fused ranks
    """
    final_rank = []
    
    # Check if all rank lists have the same length
    # Zip through all rank lists simultaneously (ranks are lists of indices)
    for r1, r2 in zip(rank1, rank2):
        # Compute the RRF score by summing reciprocal ranks for each list
        score = 1 / (k + r1) + 1/ (k+r2)  # Reciprocal rank fusion formula
        final_rank.append(score)
    
    return final_rank


class EpisodicMemoryTest:
    def __init__(self, args):
        """Initializes an empty episodic memory."""
        self.memory = []
        self.args = args

    def write(self, sentence: str, embedding: np.ndarray, infos: dict):
        """
        Inserts a new experience tuple into memory.

        Args:
            sentence (str): The text sentence.
            embedding (np.ndarray): The embedding of the sentence.
            chunk_sentence (list): The list of chunked sentences.
            chunk_embeddings (list): The list of chunked embeddings.
            steering_vectors (list): The list of steering vectors associated with the sentence.
        """
        experience = {
            "sentence": sentence,
            "embedding": embedding,
            "steering_vector": infos["steering_vector"],
            "details": []
        }
        chunk_sentence = infos['chunks']["chunk_sentence"]

        for i in range(len(chunk_sentence)):
            experience["details"].append({
                "chunk_sentence": chunk_sentence[i],
            })
        
        self.memory.append(experience)
    
    def write_chunk(self, sentence: str, embedding: np.ndarray, infos: dict):
        experience = {
            "sentence": sentence,
            "embedding": embedding,
            "steering_vector": infos["steering_vector"],
            "details": []
        }

        chunk_sentence = infos['chunks']["chunk_sentence"]
        chunk_steering_vector = infos['chunks']["chunk_steering_vector"]

        for i in range(len(chunk_sentence)):
            experience["details"].append({
                "chunk_sentence": chunk_sentence[i],
                "chunk_steering_vector": chunk_steering_vector[i]
            })

        self.memory.append(experience)

    def read(self, args, query_embedding: np.ndarray, query_text, cur_answer, distance_metric, k_chunk, k_sentence) -> List[Tuple[str, torch.Tensor, torch.Tensor, float]]:
            """
            Retrieves the top-k nearest neighbors from memory based on a distance metric.
            args: arguments
            query_embedding: The query embedding. Contains the embedding of the query sentence.
            query_text: The query text. Contains the text of the query sentence.
            cur_answer: The generated part, given the query_text.
            distance_metric: The distance metric to use for retrieval.
            k: The number of nearest neighbors to retrieve.
            """
            if not self.memory:
                return []

            # Extract embeddings from memory
            # self.consider_memory = self.memory[1:]
            embeddings = torch.stack([entry["embedding"].squeeze(0) for entry in self.memory])

            distances = []
            true_distances = []
            if distance_metric == "cosine":
                for i, emb in enumerate(embeddings):
                    d = distance.cosine(query_embedding, emb.cpu().numpy())
                    distances.append(-d)
                    true_distances.append(d)
            elif distance_metric == "euclidean":
                for emb in embeddings:                    
                    d = distance.euclidean(query_embedding, emb.cpu().numpy())
                    distances.append(-d)
                    # round up to 2 decimal places
                    true_distances.append(round(d, 2))
            else:
                raise ValueError

            sorted_indices = np.argsort(true_distances)

            # Get the sentence-level indices corresponding to the original positions
            dense_rank = np.argsort(sorted_indices)
            scores = np.array(get_bm25_scores(query_text, [self.memory[idx]["sentence"] for idx in range(len(self.memory))]))
            sorted_indices_bm25 = np.argsort(-scores)
            bm25_rank = np.argsort(sorted_indices_bm25)
            # final rank
            final_rank = np.array(rank_rrf(60, dense_rank, bm25_rank))
            # get the top k smallest ranks from the final rank
            top_k_indices = np.argsort(-final_rank)[:k_sentence]  # Sort in descending order

            # Dictionary to store the mapping of chunk to sentence index
            chunk_to_sentence_map = {sentence_idx: [chunk['chunk_sentence'] for chunk in self.memory[idx]["details"]] 
                                    for sentence_idx, idx in enumerate(top_k_indices)}

            # Flatten chunks and maintain a mapping to sentence indices
            all_chunks = []
            chunk_index_to_sentence_map = {}

            # Using enumerate to keep track of chunk_index instead of len(all_chunks)
            chunk_index = 0
            for sentence_idx, chunks in chunk_to_sentence_map.items():
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_index_to_sentence_map[chunk_index] = sentence_idx
                    chunk_index += 1

            
            if cur_answer !=  "":
                chunk_scores = np.array([get_bm25_scores(cur_answer, all_chunks)])
                sorted_indices_chunk_bm25 = np.argsort(-chunk_scores)  # Sort in descending order
                bm25_chunk_rank = np.argsort(sorted_indices_chunk_bm25)  # Convert indices to ranking            

                # merge the final_rank between sentence and chunk
                top_k_indices = np.argsort(bm25_chunk_rank)[0][:k_chunk]

                top_k_indices_steering = [chunk_index_to_sentence_map[i] for i in top_k_indices]

                steering_vector = torch.mean(
                    torch.stack([self.memory[idx]["steering_vector"] for idx in top_k_indices_steering]), 
                    dim=0
                )

            else:
                # top k indices are based on the ranking of the sentences
                steering_vector = torch.mean(
                    torch.stack([self.memory[idx]["steering_vector"] for idx in top_k_indices]), 
                    dim=0
                )

            # uncertainty = (np.mean(true_distances) - np.min(true_distances)) / (np.max(true_distances) - np.min(true_distances))
            # print("Uncertainty: ", uncertainty)
                        
            # Calculate uncertainty from true distances (ranges from 0 to 1)
            # if nan in true_distances, set uncertainty to 1
            if np.isnan(np.max(true_distances)):
                print("Nan in true distances")
                uncertainty = 1
            else:
                if np.max(true_distances) == np.min(true_distances):
                    uncertainty = 0
                else:
                    uncertainty = (np.mean(true_distances) - np.min(true_distances)) / (np.max(true_distances) - np.min(true_distances))

            return steering_vector, uncertainty


    def read_chunk(self, args, query_embedding: np.ndarray, query_text, cur_answer, distance_metric, k_chunk, k_sentence) -> List[Tuple[str, torch.Tensor, torch.Tensor, float]]:
        """
        Retrieves the top-k nearest neighbors from memory based on a distance metric.
        args: arguments
        query_embedding: The query embedding. Contains the embedding of the query sentence.
        query_text: The query text. Contains the text of the query sentence.
        cur_answer: The generated part, given the query_text.
        distance_metric: The distance metric to use for retrieval.
        k: The number of nearest neighbors to retrieve.
        """
        if not self.memory:
            return []

        # Extract embeddings from memory
        # self.consider_memory = self.memory[1:]
        embeddings = torch.stack([entry["embedding"].squeeze(0) for entry in self.memory])

        distances = []
        true_distances = []
        if distance_metric == "cosine":
            for i, emb in enumerate(embeddings):
                d = distance.cosine(query_embedding, emb.cpu().numpy())
                distances.append(-d)
                true_distances.append(d)
        elif distance_metric == "euclidean":
            for emb in embeddings:                    
                d = distance.euclidean(query_embedding, emb.cpu().numpy())
                distances.append(-d)
                # round up to 2 decimal places
                true_distances.append(round(d, 2))
        else:
            raise ValueError

        sorted_indices = np.argsort(true_distances)

        # Get the sentence-level indices corresponding to the original positions
        dense_rank = np.argsort(sorted_indices)
        scores = get_bm25_scores(query_text, [self.memory[idx]["sentence"] for idx in range(len(self.memory))])
        sorted_indices_bm25 = np.argsort(scores)
        bm25_rank = np.argsort(sorted_indices_bm25)
        # final rank
        final_rank = rank_rrf(60, dense_rank, bm25_rank)
        # get the top k scores (descending) from the final rank
        top_k_indices = np.argsort(final_rank)[-k_sentence:][::-1]  # Get top-K highest

        # Flatten chunks and maintain a mapping to chunk_embeddings
        all_chunks = []
        all_vectors = []

        for idx in top_k_indices:
            for chunk in self.memory[idx]["details"]:
                all_chunks.append(chunk["chunk_sentence"])
                all_vectors.append(chunk["chunk_steering_vector"])

        if cur_answer !=  "":
            chunk_scores = [get_bm25_scores(cur_answer, all_chunks)]
            sorted_indices_chunk_bm25 = np.argsort(-chunk_scores)  # Sort in descending order
            bm25_chunk_rank = np.argsort(sorted_indices_chunk_bm25)  # Convert indices to ranking

            # merge the final_rank between sentence and chunk
            top_k_indices = np.argsort(bm25_chunk_rank)[0][:k_chunk]


            steering_vector = torch.mean(
                torch.stack([all_vectors[idx] for idx in top_k_indices]), 
                dim=0
            )

        else:
            # top k indices are based on the ranking of the sentences
            steering_vector = torch.mean(
                torch.stack([self.memory[idx]["steering_vector"] for idx in top_k_indices]), 
                dim=0
            )

        # uncertainty = (np.mean(true_distances) - np.min(true_distances)) / (np.max(true_distances) - np.min(true_distances))
        # print("Uncertainty: ", uncertainty)
                    
        # Calculate uncertainty from true distances (ranges from 0 to 1)
        uncertainty = (np.mean(true_distances) - np.min(true_distances)) / (np.max(true_distances) - np.min(true_distances))

        return steering_vector, uncertainty


    def __len__(self):
        return len(self.memory)
    

