from diffusers import StableDiffusionPipeline
import pandas as pd
import itertools
import torch
import time
import ray
from typing import List
import click

prompt = "a photo of an astronaut riding a horse on mars"


@ray.remote
class Worker:
    def __init__(self) -> None:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.safety_checker = None
        pipe.feature_extrator = None
        pipe = pipe.to("cuda")

        @torch.compile(backend="eager")
        def inference_func(promt):
            image = pipe(prompt, num_inference_steps=10).images[0]
            return image

        self.inference_func = inference_func

        self.run_experiment(1)  # warmup run

    def is_ready(self):
        return True

    def run_experiment(self, num_iter) -> List[float]:
        duration_secs = []
        for _ in range(num_iter):
            start = time.perf_counter_ns()
            self.inference_func(prompt)
            duration_ns = time.perf_counter_ns() - start
            duration_secs.append(duration_ns / 1e9)
        return duration_secs


@click.command()
@click.option(
    "--num-iter", default=10, help="Number of iterations to run the experiment"
)
@click.option(
    "--num-workers", default=1, help="Number of workers to run the experiment"
)
@click.option(
    "--output-file", default="output.csv", help="Output file to save the results"
)
def main(num_iter, num_workers, output_file):
    ray.init()
    workers = [Worker.remote() for _ in range(num_workers)]
    ray.get([worker.is_ready.remote() for worker in workers])
    duration_secs = ray.get(
        [worker.run_experiment.remote(num_iter) for worker in workers]
    )
    duration_secs = list(itertools.chain.from_iterable(duration_secs))
    # save duration_secs to output_file
    df = pd.DataFrame(duration_secs)
    df.to_csv(output_file, index=False)
    print(df.describe())


if __name__ == "__main__":
    main()
