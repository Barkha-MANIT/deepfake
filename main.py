# main.py
import argparse
from train_model import start_training
from ui import iface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model from scratch")
    args = parser.parse_args()

    if args.train:
        start_training()
    else:
        iface.launch()
