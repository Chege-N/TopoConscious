from fastapi import FastAPI
from topoconscious.pipeline import TopoConsciousPipeline

app = FastAPI()

@app.post("/run")
def run_pipeline():
    pipe = TopoConsciousPipeline(
        bids_dir="data",
        output_dir="results"
    )
    return pipe.run()
