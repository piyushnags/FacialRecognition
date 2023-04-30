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


def create_sample_dataset(args: Any):
    loader = get_pre_loader(args)
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = get_sunet(args, device)
    model.eval()

    if args.add_noise:
        save_dir = 'blurry_dataset/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with torch.no_grad():
            for i, img in enumerate(tqdm(loader)):
                if i > args.num_blurry_batches:
                    break
                img = img.to(device)
                out = model(img)
                out = torch.clamp(out, 0, 1)
                for j in range(out.size(0)):
                    save_path = os.path.join(save_dir, f"image_{i}_{j}.png")
                    torchvision.utils.save_image(out[j], save_path)
    
    else:
        save_dir = 'sample_dataset/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, img in enumerate(tqdm(loader)):
            if i > args.num_blurry_batches:
                break
            for j in range(img.size(0)):
                save_path = os.path.join(save_dir, f"image_{i}_{j}.png")
                torchvision.utils.save_image(img[j], save_path)
    
    print("Sample dataset created!")


def save_aligned_datasets(args: Any):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    save_dir1 = 'blurry_faces/'
    save_dir2 = 'sample_faces/'

    dataset = InceptionDataset(args.root1, args.root2)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = MTCNN(256, keep_all=True, device=device)

    for i, (img1, img2) in enumerate(tqdm(loader)):
        if i > args.num_blurry_batches:
            break
        img1, img2 = img1.to(device), img2.to(device)
        out1, out2 = model(img1), model(img2)
        out1 = torch.round(out1*128+127.5)
        out2 = torch.round(out2*128+127.5)
        
        for j in range(out1.size(0)):
            path1 = os.path.join(save_dir1, f"image_{i}_{j}.png")
            path2 = os.path.join(save_dir2, f"image_{i}_{j}.png")
            torchvision.utils.save_image(out1[j], path1)
            torchvision.utils.save_image(out2[j], path2)            



def train_one_epoch(model, train_loader, device, optimizer, epoch, print_freq=10):
    model.train()

    loss_fn = nn.MSELoss()
    epoch_losses = []

    for batch_idx, (im1, im2) in enumerate(tqdm(train_loader)):
        im1, im2 = im1.to(device), im2.to(device)
        optimizer.zero_grad()

        out1 = model(im1)
        out2 = model(im2)

        loss = loss_fn(out1, out2)
        loss.backward()

        epoch_losses.append( loss.item() )
        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx} Loss, Epoch {epoch}: {loss.item():.5f} ")

        optimizer.step()
    
    avg_loss = sum(epoch_losses)/len(epoch_losses)
    print(f"Average training loss for Epoch {epoch}: {avg_loss:.5f} ")


def evaluate(model, test_loader, device, epoch):
    model.eval()

    loss_fn = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for im1, im2 in tqdm(test_loader):
            im1, im2 = im1.to(device), im2.to(device)
            out1 = model(im1)
            out2 = model(im2)
            loss = loss_fn(out1, out2)
            losses.append(loss.item())
    
    avg_loss = sum(losses)/len(losses)
    print(f"Average Evaluation Loss: {avg_loss:.5f} ")


def train(args: Any, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: Any):
    print(f"No. of Training samples: {len(train_loader)*args.batch_size}")
    print(f"No. of Validation samples: {len(test_loader)*args.batch_size}")
    
    epochs = args.num_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("No. of trainable parameters: {}".format(trainable))

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer arg')
    
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError('Invalid scheduler arg')
    
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    
    for epoch in range(1, epochs+1):
        train_one_epoch(model, train_loader, device, optimizer, epoch, args.print_freq)
        evaluate(model, test_loader, device, epoch)
        scheduler.step()
    
    torch.save(model.state_dict(), os.path.join(args.train_dir, 'inception.pth'))



if __name__ == '__main__':
    args = parse_options()
    if args.make_blurry:
        create_sample_dataset(args)
    elif args.save_aligned_ds:
        save_aligned_datasets(args)
    elif args.train:
        if args.device == 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        model = InceptionResnetV1(pretrained='vggface2', device=device)
        for child in list(model.children())[:-7]:
            for p in child.parameters():
                p.requires_grad_(False)
        train_loader, test_loader = get_loaders(args)
        train(args, model, train_loader, test_loader, device)
