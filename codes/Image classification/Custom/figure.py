import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf


def plot_figure(phase, loss, acc, prc, rec, f1, result_dir, class_names):
    prc = np.nan_to_num(prc)
    rec = np.nan_to_num(rec)
    f1 = np.nan_to_num(f1)

    if phase == 'train':
        count = 0
    if phase == 'val':
        count = 4

    plt1 = plt.figure(count + 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    plt2 = plt.figure(count + 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')

    plt3 = plt.figure(count + 3)
    plt.title('Recall ~ Precision')
    plt.xlabel('Precision')
    plt.ylabel('Recall')

    plt4 = plt.figure(count + 4)
    plt.title('F1 measure')
    plt.xlabel('Epoch')
    plt.ylabel('F1 value')

    x = range(len(acc))

    prc = list(map(list, zip(*prc)))
    rec = list(map(list, zip(*rec)))
    f1 = list(map(list, zip(*f1)))
    # Accuracy
    plt.figure(count + 1)
    plt.plot(x, loss)
    plt.figure(count + 2)
    plt.plot(x, acc)
    # Precision, Recall, F1
    for it in range(len(class_names)):
        plt.figure(count + 3)
        plt.scatter(prc[it], rec[it], label=class_names[it])
        plt.figure(count + 4)
        plt.plot(x, f1[it], label=class_names[it])
    plt3.legend(bbox_to_anchor=(0.9,0.28), fontsize='xx-small')
    plt4.legend(bbox_to_anchor=(0.9,0.28), fontsize='xx-small')


    # fig = open(result_dir + '/plot_' + '.pdf', 'w')
    # pdf = matplotlib.backends.backend_pdf.PdfPages(result_dir + '/plot_' + '.pdf')
    # for figg in range(1, 5):
    #     pdf.savefig(figg)
    # pdf.close()
    if phase == 'train':
        pdf1 = matplotlib.backends.backend_pdf.PdfPages(result_dir + '/plot_train' + '.pdf')
        for fig_num in range(1, 5):
            pdf1.savefig(fig_num)
        pdf1.close()
    if phase == 'val':
        pdf2 = matplotlib.backends.backend_pdf.PdfPages(result_dir + '/plot_val' + '.pdf')
        for fig_num in range(5, 9):
            pdf2.savefig(fig_num)
        pdf2.close()
