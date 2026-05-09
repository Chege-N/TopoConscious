# run_pipeline.py
from nilearn.datasets import fetch_spm_auditory
from topoconscious import TopoConsciousPipeline
import os

print("Downloading SPM auditory dataset...")
data_dir = fetch_spm_auditory(data_dir='./data/spm_auditory')

# The function returns a 'bunch' object with a 'dataset_dir' attribute
bids_dir = data_dir.dataset_dir
print(f"Dataset downloaded to: {bids_dir}")

# Initialize and run the pipeline with minimal parameters for a quick test
print("Initializing TopoConscious pipeline...")
pipe = TopoConsciousPipeline(
    bids_dir=bids_dir,
    output_dir="./results",
    atlas="schaefer100", # Use a reliable atlas as a fallback
    n_landmarks=40,      # Reduced landmarks for faster computation
    window_size=30,
    step=15,             # Larger step for fewer windows
    use_gpu=False
)

print("Running pipeline...")
pipe.run()
print("Pipeline finished. Check the ./results directory.")
