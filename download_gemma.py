import os
from huggingface_hub import login, snapshot_download

# Set path to persistent storage
os.environ["HF_HOME"] = "/workspace/hf_cache"
target_dir = "/workspace/Projects/PostTraining-LLM-Small/models/gemma-2-2b"

print("--- Step 1: Login ---")
# You can also use login(token="your_hf_token_here") for zero-interaction
login()

print(f"\n--- Step 2: Downloading Gemma-2-2b to {target_dir} ---")
snapshot_download(
    repo_id="google/gemma-2-2b",
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # Important: Moves actual files to /workspace
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip non-PyTorch weights
)
print("\n✅ Model downloaded successfully!")
