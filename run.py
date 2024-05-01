import warnings
import argparse
from scripts.main import main
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='save_model/Llama-2-7b-chat-hf')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--num_epochs', type=int, default=200)
    parser.add_argument('-trn', '--train_path', type=str, default='data/english-train.json')
    parser.add_argument('-val', '--val_path', type=str, default='data/english-dev.json')
    parser.add_argument('-tst', '--test_path', type=str, default='data/english-test.json')
    parser.add_argument('-sp', '--save_path', type=str, default='checkpoints')
    parser.add_argument('-r', '--r', type=int, default=16)
    parser.add_argument('-ld', '--lora_dropout', type=float, default=0.05)
    parser.add_argument('-la', '--lora_alpha', type=float, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
