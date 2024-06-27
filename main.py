import sys
from train import train
from test import test
from multi_test import multi_test
import argparse



def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--batch_size', type=int, default=128)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train', parents=[parent_parser])
    parser_train.add_argument('model_name')
    parser_train.add_argument('epochs', type=int)

    parser_test = subparsers.add_parser('test', parents=[parent_parser])
    parser_test.add_argument('model_name')

    parser_multi_test = subparsers.add_parser('multi_test', parents=[parent_parser])
    parser_multi_test.add_argument('model_names', nargs='+')

    args = parser.parse_args()



    if args.command == "train":
        train(args.model_name, args.epochs, args.batch_size)
    elif args.command == "test":
        test(args.model_name, batch_size=args.batch_size)
    elif args.command == "multi_test":
        multi_test(args.model_names, batch_size=args.batch_size)

    else:
        print("Invalid command. Use 'test' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()