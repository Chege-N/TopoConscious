from topoconscious import TopoConsciousPipeline

pipe = TopoConsciousPipeline(
    bids_dir="/home/owl/Desktop/TopoConscious/Desktop/TopoConscious",
    output_dir="./results_real",
    atlas="schaefer100",      # reliable atlas
    n_landmarks=50,
    window_size=30,
    step=5,
    use_gpu=False
)

pipe.run()
