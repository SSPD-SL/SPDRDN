import argparse

parser = argparse.ArgumentParser()

##network
parser.add_argument('--num-features', type=int, default=32)
parser.add_argument('--growth-rate', type=int, default=32)
parser.add_argument('--num-blocks', type=int, default=16)
parser.add_argument('--num-layers', type=int, default=6)

parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--stride', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)
##train
parser.add_argument('--train-file', type=str, default='data/train.h5')
parser.add_argument('--eval-file', type=str, default='data/eval.h5')
parser.add_argument('--outputs-dir', type=str, default='checkpoint')
#parser.add_argument('--weights-file', type=str)
##test
parser.add_argument('--weights-file', type=str, default='checkpoint /epoch_99.pth')
parser.add_argument('--image-file', type=str, default='data/test/input')
parser.add_argument('--result-dir', type=str, default='result')



#args = parser.parse_args()