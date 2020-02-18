# evaluate a smoothed classifier on a dataset
import argparse
import os
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from torchvision import transforms, datasets
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--p', default=2, type=int, help="p-norm for generalized Gaussian noise")
parser.add_argument('--scale_down',default=1, type=int, help="factor to scale each dimension down by")

args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    # create the smooothed classifier g
    if (args.scale_down != 1):
        base_classifier_orig = base_classifier
        base_classifier = lambda x: base_classifier_orig(torch.nn.functional.interpolate(x, scale_factor=args.scale_down))

    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma,p=args.p)

    # prepare output file
    f = open(args.outfile, 'w')
    if (args.p == 2):
        print("idx\tlabel\tpredict\tcount\tany_iid_distribution_bound\tgeneralized_gaussian_bound_over_sqrt(c)\texact_radius\tcorrect\ttime", file=f, flush=True)
    else:
        print("idx\tlabel\tpredict\tcount\tany_iid_distribution_bound\tgeneralized_gaussian_bound_over_sqrt(c)\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if (args.scale_down == 1 or args.dataset == "imagenet"):
        dataset = get_dataset(args.dataset, args.split)
    else:
        dataset = datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.Compose([
            transforms.Resize(int(32/args.scale_down)),
            transforms.ToTensor()
        ]))
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, count, radius_gen, radius_over_sqrt_c, radius_ppf  = smoothed_classifier.upper_bound_certify_generalized(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        if (args.p == 2):
            print("{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{}\t{}".format(
                i, label, prediction, count, radius_gen, radius_over_sqrt_c, radius_ppf, correct, time_elapsed), file=f, flush=True)
        else:
            print("{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{}\t{}".format(
                i, label, prediction, count, radius_gen, radius_over_sqrt_c, correct, time_elapsed), file=f, flush=True)

    f.close()
