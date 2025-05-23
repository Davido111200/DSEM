# [ACL 25 Findings] Dynamic Steering With Episodic Memory For Large Language Models
This is the codebase for the [paper](https://openreview.net/forum?id=XJIYS7s6Il&referrer=%5Bthe%20profile%20of%20Van%20Dai%20Do%5D(%2Fprofile%3Fid%3D~Van_Dai_Do1)) Dynamic Steering With Episodic Memory For Large Language Models published at Findings of ACL 2025!

![DSEM](imgs/DSEM.png)


## Prerequisites

### GYAFC Corpus
To run DSEM on the GYAFC dataset, you must first obtain permission for using the GYAFC corpus:
1. **Request permission from Yahoo**: [Yahoo Webscope L6](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l)
2. **Request permission from the authors**: [GYAFC Corpus GitHub](https://github.com/raosudha89/GYAFC-corpus)

### Environment Setup
1. Create a new Conda environment:
   ```bash
   conda create -n dsem python=3.9
   ```
2. Load and activate the environment:
   ```bash
   module load Anaconda3
   source activate
   conda activate dsem
   ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running DSEM

### Navigate to the Project Directory
```bash
cd ~/DSEM/pytorch-a2c-ppo-acktr-gail/
```

### Building Memory for DSEM
Run the following command to build memory:
```bash
python pytorch_a2c_ppo_acktr_gail/main_memory_test.py \
    --num-processes 1 \
    --num-steps 4 \
    --num-mini-batch 4 \
    --dataset paradetox \
    --prompt_version default \
    --exemplar_method random \
    --model_type gemma-9b \
    --model_size 9b \
    --batch_size 1 \
    --gpus 0 \
    --in_8bit True \
    --seed 0 \
    --n_memory_samples 100
```

### Running Inference with DSEM
Run the following command for inference:
```bash
python pytorch_a2c_ppo_acktr_gail/evaluate_memory_test.py \
    --num-processes 1 \
    --num-steps 4 \
    --log-interval 1 \
    --dataset paradetox \
    --prompt_version default \
    --model_type gemma-9b \
    --model_size 9b \
    --batch_size 1 \
    --gpus 0 \
    --in_8bit True \
    --seed 0 \
    --metric toxicity \
    --n_memory_samples 100 \
    --distance_metric cosine \
    --mix_icv True \
    --mix_strat soft \
    --chunk_size 4 \
    --n_neighbors 9 \
    --n_neighbor_chunks 9
```

