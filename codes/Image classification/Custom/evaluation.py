import numpy as np


def evaluation(datasets_sizes, running_loss, phase, table):

    # Preparation
    TP = np.diagonal(table)
    FP = np.sum(table, 0)-TP
    FN = np.sum(table, 1)-TP

    # Calculation
    loss = running_loss / datasets_sizes[phase]
    acc = np.sum(TP)/datasets_sizes[phase]
    prec = np.array(TP)/(np.array(TP)+np.array(FP))
    recall = np.array(TP)/(np.array(TP)+np.array(FN))
    F1 = 2*prec*recall/(prec+recall)

    # Output
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    # for it in range(classes):
    #     print('{}: Precision = {:.4f}, Recall = {:.4f}, F1 measure = {:.4f}'.format(
    #         class_names[it], prec[it], recall[it], F1[it]))

    return [loss, acc, prec, recall, F1]
