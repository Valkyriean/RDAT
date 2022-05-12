import argparse
import sys
from sklearn.metrics import precision_recall_fscore_support

#parameters
debug = False

###########
#functions#
###########

def convert_label(label):
    if label == "rumour":
        return 1
    elif label == "nonrumour":
        return 0
    else:
        raise Exception("label classes must be 'rumour' or 'nonrumour'")

######
#main#
######

def main(args):

    try:
        groundtruth = [item.strip() for item in open(args.groundtruth, "r").readlines()]
        predictions = [item.strip() for item in open(args.predictions, "r").readlines()]

        if (len(groundtruth)!=len(predictions)):
            raise Exception("groundtruth file must have the same number of lines as the predictions file")

        y_true, y_pred = [], []

        for vi, v in enumerate(groundtruth):
            y_pred.append(convert_label(predictions[vi]))
            y_true.append(convert_label(v))

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary")
    except Exception as error:
        print("Error:", error)
        raise SystemExit


    print("Performance on the rumour class:")
    print("Precision =", p)
    print("Recall    =", r)
    print("F1        =", f)
        

if __name__ == "__main__":

    #parser arguments
    desc = "Computes precision, recall and F-score of the rumour class"
    parser = argparse.ArgumentParser(description=desc)

    #arguments
    parser.add_argument("--predictions", required=True, help="text file containing system predictions (one line per label)")
    parser.add_argument("--groundtruth", required=True, help="text file containing ground truth labels (one line per label)")
    args = parser.parse_args()

    main(args)
