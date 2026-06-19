# train/AlphaRec.py

import os
import json
from sklearn.decomposition import PCA
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.data.dataset import Dataset

class AlphaRec(GeneralRecommender):
    """
    An example RecBole-adapted AlphaRec model.
    Note that this version inherits from RecBole's GeneralRecommender to comply with its configuration
    and training loop, rather than directly subclassing nn.Module.
    """
    def __init__(self, config, dataset):
        # Initialize using RecBole's config and dataset.
        super(AlphaRec, self).__init__(config, dataset)

        # Retrieve configuration parameters
        self.tau = config['tau']
        self.embed_size = config['embed_size']
        self.lm_model = config['semantic_encoder']  # This could be the identifier or an encoder instance.
        
        # for full‐sort caching
        self._cached_user_emb = None   # will hold [num_users, D]
        self._cached_item_emb = None   # will hold [num_items, D]
        
        self.init_item_cf_embeds = torch.tensor(dataset.item_cf_embeds, dtype=torch.float32, device=self.device)
        
        # 1) pull out NumPy array of item embeddings
        item_embs = dataset.item_cf_embeds # shape: (num_items, embed_dim)

        # 2) get train‐split user/item pairs
        train_users = dataset.inter_feat[self.USER_ID].numpy()
        train_items = dataset.inter_feat[self.ITEM_ID].numpy()

        # 3) build a list of items per user
        user2items = [[] for _ in range(dataset.user_num)]
        for u, i in zip(train_users, train_items):
            user2items[int(u)].append(int(i))
        
        # 4) average
        user_cf_embs = np.zeros((dataset.user_num, item_embs.shape[1]),
                                 dtype=item_embs.dtype)
        for u, items in enumerate(user2items):
            if items:  # if the user has any training interactions
                user_cf_embs[u] = np.mean(item_embs[items], axis=0)

        # 5) wrap as a frozen tensor
        self.init_user_cf_embeds = torch.tensor(
            user_cf_embs, dtype=torch.float32, device=self.device
        )

        self.init_embed_shape = self.init_user_cf_embeds.shape[1]
        
        # A simple linear transformation as in your original mlp (without nonlinearity)
        self.mlp = nn.Linear(self.init_embed_shape, self.embed_size, bias=False)

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self._cached_user_emb is not None or self._cached_item_emb is not None:
            self._cached_user_emb, self._cached_item_emb = None, None

        users     = interaction[self.USER_ID]     # [B]
        pos_items = interaction[self.ITEM_ID]     # [B]
        B = users.size(0)

        # 1) compute & normalize all embeddings once
        norm_all_user_emb = F.normalize(self.mlp(self.init_user_cf_embeds), dim=-1)  # [U, D]
        norm_all_item_emb = F.normalize(self.mlp(self.init_item_cf_embeds), dim=-1)  # [I, D]

        # 2) pick out this batch’s user & positive embeddings
        curr_user_emb = norm_all_user_emb[users]     # [B, D]
        curr_pos_item_emb = norm_all_item_emb[pos_items] # [B, D]

        # 3) Compute in batch user-item similarity matrix [B, B]
        logits = torch.matmul(curr_user_emb, curr_pos_item_emb.transpose(0, 1))
        logits = logits / self.tau

        # 4) Assign targets for in-batch contrastive learning
        targets = torch.arange(B, device=logits.device) 

        # 5) Calculate InfoNCE loss
        loss = F.cross_entropy(logits, targets)

        return loss
    
    def predict(self, interaction):
        users = interaction[self.USER_ID]      # e.g. tensor([u1,u2,...])
        items = interaction[self.ITEM_ID]      # e.g. tensor([i1,i2,...])
        
        all_users = self.mlp(self.init_user_cf_embeds)
        all_items = self.mlp(self.init_item_cf_embeds)
        
        u_emb = F.normalize(all_users[users], dim=-1)
        i_emb = F.normalize(all_items[items], dim=-1)
        return torch.sum(u_emb * i_emb, dim=-1)  # shape: [batch_size]

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        Full-sort prediction method for ranking all items for each user.
        This method is invoked during evaluation.
        """
        # Extract user indices from the interaction object (e.g., a tensor)
        users = interaction[self.USER_ID]  # Expected shape: [batch_size]
        
        # 1) on first call, build & cache the per‐user and per‐item embeddings
        if self._cached_user_emb is None or self._cached_item_emb is None:
            # project your CF init embeddings
            all_users = self.mlp(self.init_user_cf_embeds)  # [U, D]
            all_items = self.mlp(self.init_item_cf_embeds)  # [I, D]

            # normalize once
            self._cached_user_emb = F.normalize(all_users, dim=-1)
            self._cached_item_emb = F.normalize(all_items, dim=-1)

        # 2) lookup only the users in this batch
        batch_users = self._cached_user_emb[users]        # [B, D]

        # 3) score against every item
        scores = batch_users @ self._cached_item_emb.t()  # [B, I]
        
        return scores
    
class AlphaRecDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        self.item_cf_embeds = self.load_plm_embedding()

    def load_plm_embedding(self):
        # Try metadata cache first (for shared embeddings), then fall back to data_path
        feat_path_metadata = os.path.join("cache", "metadata", self.dataset_name, f'{self.dataset_name}.{self.plm_suffix}')
        feat_path_local = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        
        if os.path.exists(feat_path_metadata):
            feat_path = feat_path_metadata
        else:
            feat_path = feat_path_local
        
        loaded_feat = np.load(feat_path, allow_pickle=True).reshape(-1, self.plm_size)
        assert loaded_feat.shape == (self.item_num - 1, self.plm_size), f"Loaded feature shape {loaded_feat.shape} does not match expected shape {(self.item_num - 1, self.plm_size)}"

        data_maps_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.data_maps')
        with open(data_maps_path, 'r') as f:
            data_maps = json.load(f)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(data_maps['item2id'][token]) - 1]
        return mapped_feat
    
class HyperParamLoader:
    """
    Helper class: generates all hyperparameter combinations from a dictionary (arg_range)
    using a DFS-based approach.
    """
    def __init__(self, arg_range):
        self.arg_range = arg_range
        self.k_list = list(arg_range.keys())
        self.choice = np.zeros((len(self.k_list)), dtype=int)
        self.args = []
        self._dfs(0)
        # Sanity-check: total combinations equals product of all list lengths
        assert len(self.args) == np.prod([len(v) for v in arg_range.values()])
        self.n_args = len(self.args)
        print('Total hyperparameter combinations:', self.n_args, flush=True)
    
    def _dfs(self, layer):
        if layer == len(self.k_list):
            ans = {}
            for l, k in enumerate(self.k_list):
                ans[k] = self.arg_range[k][self.choice[l]]
            self.args.append(ans)
            return
        
        k = self.k_list[layer]
        for i in range(len(self.arg_range[k])):
            self.choice[layer] = i
            self._dfs(layer + 1)
    
class CFBaseTask:
    def __init__(self, dataset_name, gpu_id=0, cache_path="./cache", enc_batch_size=32, eval_batch_size=8, **kwargs):
        self.dataset_name = dataset_name
        self.results = {}
        self.cache_path = os.path.join(cache_path, "cf")
        self.enc_batch_size = enc_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = init_device(gpu_id)
        self.features_needed = kwargs.get('features_needed', ['title'])
    
    def load_data(self, semantic_encoder=None):
        amazon2023 = ['All_Beauty', 'Video_Games', 'Baby_Products']

        # Check if task-specific data files (.inter files and data_maps) exist
        task_data_dir = os.path.join(self.cache_path, self.dataset_name)
        data_maps_file = os.path.join(task_data_dir, f"{self.dataset_name}.data_maps")
        train_inter_file = os.path.join(task_data_dir, f"{self.dataset_name}.train.inter")

        # Check if embedding files exist
        processed_dir_metadata = os.path.join("cache", "metadata", self.dataset_name, f'{self.dataset_name}.{semantic_encoder.name}.npy')
        processed_dir_local = os.path.join(self.cache_path, self.dataset_name, f'{self.dataset_name}.{semantic_encoder.name}.npy')
        embeddings_exist = os.path.exists(processed_dir_metadata) or os.path.exists(processed_dir_local)

        # Always process data if task-specific files or embeddings don't exist
        if not (os.path.exists(data_maps_file) and os.path.exists(train_inter_file) and embeddings_exist):
            print(f"[CFBaseTask] Processing data for {self.dataset_name} ...")
            if self.dataset_name in amazon2023:
                # Now call process_amazon(...) directly
                # TODO: can adjust the number of embeddings feature, need to derive how this is set
                emb_size = 64
            else:
                assert NotImplementedError()
        else:
            print(f"[CFBaseTask] Data for {self.dataset_name} is already processed.")

        # Find and load embeddings - try metadata cache first, then data cache
        if os.path.exists(processed_dir_metadata):
            processed_dir = processed_dir_metadata
        else:
            processed_dir = processed_dir_local

        features = np.load(processed_dir)
        print(f"Loaded raw embeddings shape: {features.shape}")

        # Apply PCA post-processing if needed
        pca_config = semantic_encoder.get_pca_config()
        features, processed_emb_path = apply_pca_if_needed(
            features,
            semantic_encoder.name,
            self.dataset_name,
            pca_config
        )
        print(f"Final embeddings shape after PCA: {features.shape}")

        self.emb_size = features.shape[-1]
        # Store processed emb path for training
        self.processed_emb_path = processed_emb_path
    
    def run(self, semantic_encoder, gpu_id=0, hyperdict={"learning_rate": [3e-3, 1e-3, 3e-4]}):
        """
        Performs hyperparameter tuning by iterating over all combinations provided
        in hyperdict. For each combination, it calls run_single() and logs progress.
        The same log file (self.log_file) will be used as the final result file.
        """
        self.model_name = semantic_encoder.name
        print(f"[CFBaseTask] Starting hyperparameter tuning with model={semantic_encoder.name}, domain={self.dataset_name}")
        
        # Setup logging: create 'hyper_cf' folder and log file.
        log_dir = "hyper_cf"
        os.makedirs(log_dir, exist_ok=True)
        # Generate a log file name based on sys.argv
        self.log_file = os.path.join(log_dir, f"log_{int(time.time())}.txt")
        hlog = open(self.log_file, 'a+')
        
        # Write meta information
        hlog.write("======= META =======\n")
        meta_info = {
            "model_name": semantic_encoder.name,
            "domain_name": self.dataset_name,
            "hyperdict": hyperdict
        }
        hlog.write(str(meta_info) + "\n\n")
        hlog.flush()
        
        # Base configuration to be passed to run_single()
        pca_config = semantic_encoder.get_pca_config()
        suffix = self.processed_emb_path.rsplit('/', 1)[-1]
        suffix = '.'.join(suffix.split('.')[1:])

        base_config = {
            "gpu_id": gpu_id,
            "data_path": self.cache_path,
            "device": self.device,
            "plm_size": self.emb_size,
            "plm_suffix": suffix,
            "semantic_encoder": semantic_encoder.name.split("/")[-1]
        }
        
        # Instantiate hyperparameter loader to generate all combinations
        hp_loader = HyperParamLoader(hyperdict)
        
        best_valid_score = None
        best_params = None
        best_result_dict = None
        best_round = None
        
        # Iterate over each hyperparameter combination
        for idx, hyper_params in enumerate(tqdm(hp_loader.args, desc="Hyperparameter Tuning")):
            hlog.write(f"======= Round {idx} =======\n")
            hlog.write(str(hyper_params) + "\n\n")
            hlog.flush()
            
            # Merge base configuration with current hyperparameter combination
            config_overrides = base_config.copy()
            config_overrides.update(hyper_params)
            
            print(f"[CFBaseTask] Tuning round {idx+1}/{hp_loader.n_args} with hyperparameters: {hyper_params}")
            # Call run_single() with the current configuration
            _, _, result_dict = run_single(
                model_name="AlphaRec",
                dataset=self.dataset_name,
                pretrained_file="",
                **config_overrides
            )
            
            hlog.write("Best Valid Result: " + str(result_dict["best_valid_result"]) + "\n\n")
            hlog.flush()
            
            current_valid_score = result_dict["best_valid_score"]
            print(f"[CFBaseTask] Round {idx+1}: Best Valid Score = {current_valid_score}")
            
            # Update the best configuration if current run is better
            if best_valid_score is None or current_valid_score > best_valid_score:
                hlog.write(f"\n Best Valid Updated: {best_valid_score} -> {current_valid_score}\n\n")
                hlog.flush()
                best_valid_score = current_valid_score
                best_params = hyper_params
                best_result_dict = result_dict
                best_round = idx
        
        # Write final summary
        hlog.write("======= FINAL =======\n")
        hlog.write(f"Best Round: {best_round}\n")
        hlog.write(f"Best Valid Score: {best_valid_score}\n")
        hlog.write("Best Params: " + str(best_params) + "\n")
        hlog.write("Final Valid Result: " + str(best_result_dict.get("best_valid_result", "")) + "\n")
        hlog.write("Final Test Result: " + str(best_result_dict.get("test_result", "")) + "\n")
        hlog.close()
        
        # Save the best results into the task's results (if needed for other purposes)
        self.results["best_valid_score"] = best_valid_score
        self.results["best_params"] = best_params
        if best_result_dict:
            self.results["best_valid_result"] = best_result_dict["best_valid_result"]
            self.results["test_result"] = best_result_dict["test_result"]
        
        print("[CFBaseTask] Hyperparameter tuning completed.")
        print(f"Best Valid Score: {best_valid_score}")
        print(f"Best Hyperparameters: {best_params}")

    def save_results(self, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        result_file = os.path.join(output_folder, f"cf_{self.model_name}_{self.dataset_name}_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
            
    def __str__(self):
        return f"CFTask(domain={self.dataset_name})"
    
# Helper functions to adapt AlphaRec
def init_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS).")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU.")
    return device

def init_semantic_encoder(model, gpu_id, batch_size, **kwargs):
    raise NotImplementedError("Semantic encoder initialization is not implemented. This is a placeholder function.")
    # raise ValueError(f'Unknown semantic encoder name {model}.')

    # print(f'Suffix of the semantic encoder: {encoder.name}')
    # return encoder
    
def apply_pca_if_needed(embeddings: np.ndarray,
                       semantic_encoder_name: str,
                       dataset_name: str,
                       pca_config: dict,
                       cache_dir: str = "./cache/metadata") -> Tuple[np.ndarray, Optional[str]]:
    """
    Convenience function to apply PCA if needed.
    
    Args:
        embeddings: Raw embeddings
        semantic_encoder_name: Name of the semantic encoder (for caching)
        dataset_name: Name of the dataset (for caching)
        pca_config: Dict with keys 'enabled', 'n_components', 'whiten'
        cache_dir: Cache directory for PCA files
        
    Returns:
        Tuple of (processed embeddings, cache path)
    """
    os.makedirs(os.path.join(cache_dir, dataset_name), exist_ok=True)

    n_components = int(pca_config.get('n_components', 0) or 0)
    whiten = bool(pca_config.get('whiten', False))
    force_recompute = bool(pca_config.get('force_recompute', False))
    pca_flag = (
        pca_config.get('enabled', False)
        and 0 < n_components < embeddings.shape[1]
    )

    pca_suffix = f"-pca-d{n_components}-w{whiten}" if pca_flag else ""
    processed_emb_path = os.path.join(
        cache_dir,
        dataset_name,
        f"{dataset_name}.{semantic_encoder_name}{pca_suffix}.npy")

    if not pca_flag:
        # No PCA needed or invalid component count
        return embeddings, processed_emb_path

    if not force_recompute and os.path.exists(processed_emb_path):
        try:
            cached_embeddings = np.load(processed_emb_path)
            if cached_embeddings.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    f"Cached embeddings size mismatch {cached_embeddings.shape} vs {embeddings.shape}"
                )
            print(f"Loading cached PCA embeddings from {processed_emb_path}")
            return cached_embeddings, processed_emb_path
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}")

    print(f"Fitting PCA with n_components={n_components}, whiten={whiten}")
    pca_model = PCA(n_components=n_components, whiten=whiten)
    pca_model.fit(embeddings)

    processed_embeddings = pca_model.transform(embeddings)
    np.save(processed_emb_path, processed_embeddings)
    print(f"Saved PCA-processed embeddings to {processed_emb_path}")

    return processed_embeddings, processed_emb_path