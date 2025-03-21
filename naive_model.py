# Naive model
def naive_model(all_labels):
    preds = [0 for i in range(len(all_labels))]
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(all_labels)):
        tp += ((preds[i] == 1) & (all_labels[i] == 1)).sum().item()
        fp += ((preds[i] == 1) & (all_labels[i] == 0)).sum().item()
        tn += ((preds[i] == 0) & (all_labels[i] == 0)).sum().item()
        fn += ((preds[i] == 0) & (all_labels[i] == 1)).sum().item()

    acc = (tp + tn) / (tp + tn +fp + fn) if (tp + tn +fp + fn) > 0 else 0  
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return acc, prec, rec, f1