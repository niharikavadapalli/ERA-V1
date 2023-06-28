import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def plot_misclassified_images(incorrect_examples, incorrect_pred,  incorrect_labels):
    fig,ax = plt.subplots(3,5)
    ax = ax.ravel()
    for i in range(15):
        image = incorrect_examples[0][i]
        t = torch.from_numpy(image)
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.24703223, 1/0.24348513, 1/0.26158784 ]),
                                    transforms.Normalize(mean = [ -0.49139968, -0.48215841, -0.44653091 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
        inv_tensor = invTrans(t)
        ax[i].imshow(inv_tensor.permute(1,2,0))
        ax[i].set_title(f"{incorrect_pred[0][i]}/{incorrect_labels[0][i]}")
        ax[i].set( xticks=[], yticks=[])
    plt.axis('off')
    plt.tight_layout()
    plt.xlabel('Misclassified Images')
    plt.ylabel('Loss')
    plt.legend("Incorrect Prediction/Actual Label")
    plt.show()

    
    
def plot_network_performance(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")