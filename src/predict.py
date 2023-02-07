import os
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import get_model
from data import FloatingSeaObjectDataset
from visualization import plot_batch
from transforms import get_transform
import json
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, jaccard_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data/data/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for train/test region split"
    )
    parser.add_argument("--workers", type=int, default=0)
    #parser.add_argument(
    #    "--augmentation-intensity",
    #    type=int,
    #    default=1,
    #    help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)",
    #)
    parser.add_argument("--pretrained_model_path", type=str, 
                        default="/data/floatingobjects/floatingobjects/" + \
                        "src/models/unet-model_0.pth.tar")
    parser.add_argument("--add_fdi_ndvi", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    #parser.add_argument("--epochs", type=int, default=1)
    # parser.add_argument("--no-pretrained", action="store_true")
    #parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tensorboard_logdir", type=str, default=None)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1,
        help="positional weight for the floating object class, large values counteract",
    )
    args = parser.parse_args()
    # args.image_size = (args.image_size,args.image_size)
    return args


def main(args):
    data_path = args.data_path

    batch_size = args.batch_size
    workers = args.workers
    image_size = args.image_size
    device = args.device
    #n_epochs = args.epochs
    #learning_rate = args.learning_rate

    tensorboard_logdir = args.tensorboard_logdir

    dataset = FloatingSeaObjectDataset(
        data_path,
        fold="test",
        transform=get_transform(
            "test",
            add_fdi_ndvi=args.add_fdi_ndvi,
        ),
        output_size=image_size,
        seed=args.seed,
        hard_negative_mining=False,
    )

    # store run arguments in the same folder
    run_arguments = vars(args)
    run_arguments["test_regions"] = ", ".join(dataset.regions)

    print(run_arguments)

    # loading training datasets
    test_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    # compute the number of labels in each class
    # weights = compute_class_occurences(train_loader) #function that computes the occurences of the classes
    #pos_weight = torch.FloatTensor([float(args.pos_weight)]).to(device)

    #bcecriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    #def criterion(y_pred, target, mask=None):
    #    """a wrapper around BCEWithLogitsLoss that ignores no-data
    #    mask provides a boolean mask on valid data"""
    #    loss = bcecriterion(y_pred, target)
    #    if mask is not None:
    #        return (loss * mask.double()).mean()
    #    else:
    #        return loss.mean()
        
    modelpath = args.pretrained_model_path

    model = get_model(
        os.path.basename(modelpath).split("-")[0].lower(), 
        inchannels=12 if not args.add_fdi_ndvi else 14,
    )
    # self.model = get_model(modelname, inchannels=12 if not add_fdi_ndvi else 14)
    model.load_state_dict(
        torch.load(modelpath, map_location=torch.device(device))["model_state_dict"]
    )
    model = model.to(device)
    #transform = get_transform("test", add_fdi_ndvi=args.add_fdi_ndvi)
    #use_test_aug = use_test_aug #TODO

    
    #start_epoch = 1
    logs = []

    # create summary writer if tensorboard_logdir is not None
    writer = (
        SummaryWriter(log_dir=tensorboard_logdir)
        if tensorboard_logdir is not None
        else None
    )

    #for epoch in range(start_epoch, n_epochs + 1):
        #trainloss = training_epoch(model, train_loader, optimizer, criterion, device)
    metrics = testing_epoch(model, test_loader, device, writer) #model, test_loader, criterion, device, writer)

    log = dict(
        #epoch=epoch,
        #trainloss=trainloss,
        #valloss=valloss,
        metrics=metrics
    )
    log.update(metrics)

    logs.append(log)
    #idx_tensorboard = 0
    

    # retrieve best loss by iterating through previous logged losses
    #best_loss = min([l["valloss"] for l in logs])
    best_kappa = max([l["kappa"] for l in logs])
    kappa = metrics["kappa"]

    #msg_loss = ""  # write save model message in the same line of the pring
    #if valloss <= best_loss:
    #    msg_loss = (
    #        f"lowest loss is {valloss}" 
    #    )
    if kappa >= best_kappa:
        msg_kappa = (
            f"highest kappa is {best_kappa}" 
        )

    metrics_message = ", ".join([f"{k} {v:.2f}" for k, v in metrics.items()])

    print(
        f"{metrics_message}, {msg_kappa}"
    )


def testing_epoch(model, test_loader, device, writer): #criterion, device, writer):
    with torch.no_grad():
        model.eval()
        #losses = []
        metrics = dict(precision=[], recall=[], fscore=[], kappa=[], mIoU=[])
        with tqdm(enumerate(test_loader), total=len(test_loader), leave=False) as pbar:
            for idx, batch in pbar:
                im, target, id = batch
                im = im.to(device)
                target = target.to(device)
                y_pred = model(im)
                #valid_data = im.sum(1) != 0  # all pixels > 0
                #loss = criterion(y_pred.squeeze(1), target, mask=valid_data)
                #losses.append(loss.cpu().detach().numpy())
                #pbar.set_description(f"test loss {np.array(losses).mean():.4f}")
                predictions = (y_pred.exp() > 0.5).cpu().detach().numpy()
                y_true = target.cpu().view(-1).numpy().astype(bool)
                y_pred = predictions.reshape(-1)
                p, r, f, s = precision_recall_fscore_support(
                    y_true=y_true, y_pred=y_pred, zero_division=0
                )
                metrics["kappa"].append(cohen_kappa_score(y_true.astype(int), y_pred.astype(int)))
                metrics["mIoU"].append(jaccard_score(
                    y_true.astype(int), y_pred.astype(int), pos_label=1, average="binary"))
                metrics["precision"].append(p)
                metrics["recall"].append(r)
                metrics["fscore"].append(f)
                
                if writer is not None:
                    #writer.add_scalars(
                    #    "loss", {"train": trainloss, "val": valloss}, global_step=epoch
                    #)
                    fig = predict_images(batch, model, device)
                    writer.add_figure("predictions_test", fig, global_step=idx) #, global_step=epoch)

                    predictions, targets = get_scores(batch, model, device)
                    targets = targets.reshape(-1)
                    targets = targets > 0.5  # make to bool
                    predictions = predictions.reshape(-1)
                    #writer.add_pr_curve("unbalanced", targets, predictions, global_step=1) #, global_step=epoch)

                    # make predictions and targets balanced by removing not floating pixels until numbers of positive
                    # and negative samples are equal
                    floating_predictions = predictions[targets]
                    not_floating_predictions = predictions[~targets]
                    np.random.shuffle(not_floating_predictions)
                    not_floating_predictions = not_floating_predictions[
                        : len(floating_predictions)
                    ]
                    predictions = np.hstack([floating_predictions, not_floating_predictions])
                    targets = np.hstack(
                        [
                            np.ones_like(floating_predictions),
                            np.zeros_like(not_floating_predictions),
                        ]
                    )
                    #writer.add_pr_curve("balanced", targets, predictions, global_step=1) #, global_step=epoch)
                

    for k, v in metrics.items():
        metrics[k] = np.array(v).mean()
        
    
    

    return metrics #np.array(losses).mean(), metrics


def predict_images(batch, model, device): #val_loader, model, device):
    images, masks, id = batch #next(iter(val_loader))
    N = images.shape[0]

    # plot at most 5 images even if the batch size is larger
    if N > 5:
        images = images[:5]
        masks = masks[:5]

    logits = model(images.to(device)).squeeze(1)
    y_preds = torch.sigmoid(logits).detach().cpu().numpy()
    return plot_batch(images, masks, y_preds)


def get_scores(batch, model, device, n_batches=5): #val_loader, model, device, n_batches=5):
    y_preds = []
    targets = []
    with torch.no_grad():
        for i in range(n_batches):
            images, masks, id = batch #next(iter(val_loader))
            logits = model(images.to(device)).squeeze(1)
            y_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(masks.detach().cpu().numpy())
    return np.vstack(y_preds), np.vstack(targets)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    