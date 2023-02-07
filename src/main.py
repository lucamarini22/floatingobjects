import argparse
from train import main as train
from predict import main as predict
import os
import pathlib
from utils import get_today_str, get_project_path


def main(args):
    if args.mode == "train":
        for seed in range(5):
            args.seed = seed

            args.today_str = get_today_str()
            proj_path = get_project_path()
            args.model_id = f"{args.model}-model-{seed}-{args.today_str}"
            # path of new or pre-existing trained model
            args.snapshot_path = os.path.join(
                proj_path, args.results_dir, f"{args.model_id}.pth.tar"
            )
            # path of folder to store model
            args.model_folder = os.path.join(proj_path, args.results_dir, args.model_id)
            # path where to store files for Tensorboard
            args.tensorboard_logdir = os.path.join(
                proj_path, args.tensorboard, args.model_id
            )
            # create folders to store tensorboard files
            os.makedirs(args.model_folder, exist_ok=True)
            os.makedirs(args.tensorboard_logdir, exist_ok=True)
            # start training
            train(args)
    elif args.mode == "predict":
        predict(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "predict"])
    parser.add_argument("--results-dir", type=str, default="models")
    parser.add_argument("--tensorboard", type=str, default="log")

    # train arguments
    parser.add_argument("--data-path", type=str, default="/data")
    parser.add_argument("--batch-size", type=int, default=256)  # 8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--augmentation-intensity",
        type=int,
        default=1,
        help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)",
    )
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--add-fdi-ndvi", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    # parser.add_argument('--tensorboard-logdir', type=str, default=None)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1,
        help="positional weight for the floating object class, large values counteract",
    )
    parser.add_argument(
        "--hard_negative_mining_train_dataset", type=bool, default=False
    )
    parser.add_argument("--loss", type=str, default="bce")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_args())
