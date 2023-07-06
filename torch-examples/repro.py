from torchvision.models import resnet18
import torch
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

dynamo_config.output_graph_code = True
dynamo_config.output_code = True
inductor_config.debug = True
model = resnet18(pretrained=True).eval().cuda()
inp = torch.zeros((1, 3, 224, 224)).cuda()


@torch.compile()
@torch.no_grad()
def inference_func(inp):
    return model(inp)


image = inference_func(inp)
