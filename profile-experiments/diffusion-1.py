from diffusers import StableDiffusionPipeline
import threading
import pandas as pd
import itertools
import torch
import time
import ray
from typing import List
import click
from fijit_py import Fijit

prompt = "a photo of an astronaut riding a horse on mars"


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
    "--num-streams", default=1, help="Number of cuda streams to run the experiment"
)
@click.option(
    "--output-file", default="output.csv", help="Output file to save the results"
)
def main(num_iter, num_streams, output_file):
    # Fijit(enable_activity_api=True, enable_callback_api=False).run()

    barrier = threading.Barrier(num_streams)

    def thread_func(result_list):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            worker = Worker()
            barrier.wait()
            duration_secs = worker.run_experiment(num_iter)
        result_list.append(duration_secs)

    result_list = []
    threads = [
        threading.Thread(target=thread_func, args=(result_list,))
        for _ in range(num_streams)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    duration_secs = list(itertools.chain.from_iterable(result_list))

    df = pd.DataFrame(duration_secs)
    df.to_csv(output_file, index=False)
    print(df.describe())


if __name__ == "__main__":
    main()
