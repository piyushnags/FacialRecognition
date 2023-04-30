from typing import Any
from tqdm import tqdm
from train_utils import *
from facenet_pytorch import *
from models import SUNet_model
from utils import get_options, load_checkpoint, parse_options


def get_sunet(args: Any, device: Any) -> SUNet_model:
    options = get_options(args.yaml_path)
    model = SUNet_model(options)
    model.to(device)
    load_checkpoint(model, args.weights, device)
    return model


def create_blurry_dataset(args: Any):
    loader = get_pre_loader(args)
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = get_sunet(args, device)
    model.eval()

    save_dir = 'blurry_dataset/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():
        for i, img in enumerate(tqdm(loader)):
            img = img.to(device)
            out = model(img)
            out = torch.clamp(out, 0, 1)
            for j in range(out.size(0)):
                torchvision.utils.save_image(out[j], f"image_{i}_{j}.png")
    
    print("Blurry dataset created!")



if __name__ == '__main__':
    args = parse_options()
    if args.make_blurry:
        create_blurry_dataset(args)
