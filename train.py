import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, ResizerFixed224
from torch.utils.data import DataLoader, Subset

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')

    parser.add_argument('--depth', help='Resnet depth', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--resume', help='Checkpoint path', default=None)

    parser = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO')

        full_dataset = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), ResizerFixed224()]))
        dataset_train = Subset(full_dataset, list(range(500)))

        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), ResizerFixed224()]))
    else:
        raise ValueError('Dataset type not understood')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=16, drop_last=False)

    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=16, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    start_epoch = 0

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=full_dataset.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=full_dataset.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=full_dataset.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=full_dataset.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=full_dataset.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    retinanet = retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet)

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    if parser.resume is not None:

        print("Loading checkpoint:", parser.resume)

        checkpoint = torch.load(parser.resume, map_location=device)

        retinanet.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1

        print("Resumed from epoch:", start_epoch)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:

                optimizer.zero_grad()

                imgs = data['img'].to(device).float()
                annots = data['annot']

                classification_loss, regression_loss = retinanet([imgs, annots])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)

                optimizer.step()

                loss_value = loss.detach().item()

                loss_hist.append(loss_value)
                epoch_loss.append(loss_value)

                if (iter_num % 100) == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, classification_loss.item(), regression_loss.item(), np.mean(loss_hist)))

            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        torch.save(
            {
                'epoch': epoch_num,
                'model_state_dict': retinanet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num)
        )

    retinanet.eval()

    torch.save(retinanet.module.state_dict(), 'model_final.pt')


if __name__ == '__main__':
    main()