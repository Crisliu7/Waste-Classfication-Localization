import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualization(model, dataloaders, use_gpu, class_names, result_dir, pic_num=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()
    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(pic_num // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]), fontsize='small')
            imshow(inputs.cpu().data[j])
            if images_so_far == pic_num:
                model.train(mode=was_training)
                # save as a pdf
                fig = open(result_dir + '/model_sample.pdf', 'w')
                plt.savefig(result_dir + '/model_sample.pdf')
                fig.close()
                return
    model.train(mode=was_training)
