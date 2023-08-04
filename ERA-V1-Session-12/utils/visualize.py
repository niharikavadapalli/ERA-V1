import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchinfo

def get_incorrect_predictions(model, test_loader, device):
    incorrect_examples = []
    incorrect_pred = []  
    incorrect_labels = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            if not pred.eq(target.view_as(pred)).item():
                incorrect_examples.append(data)
                incorrect_pred.append(pred)
                incorrect_labels.append(target)
    return incorrect_examples, incorrect_pred,  incorrect_labels

def plot_misclassified_images(model, test_loader, device, class_names, rows=3, columns=5):
    incorrect_examples, incorrect_pred,  incorrect_labels = get_incorrect_predictions(model, test_loader, device)
    fig,ax = plt.subplots(rows,columns)
    ax = ax.ravel()
    for i in range(rows*columns):
        image = incorrect_examples[0][i]
        t = torch.from_numpy(image)
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.24703223, 1/0.24348513, 1/0.26158784 ]),
                                    transforms.Normalize(mean = [ -0.49139968, -0.48215841, -0.44653091 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
        inv_tensor = invTrans(t)
        ax[i].imshow(inv_tensor.permute(1,2,0))
        ax[i].set_title(f"{class_names[incorrect_pred[0][i]]}/{class_names[incorrect_labels[0][i]]}")
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
    

def print_samples(loader, count=16):
    """
    Print samples input images
    """
    # Print Random Samples
    if not count % 8 == 0:
        return
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in loader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8), 8, i + 1, xticks=[], yticks=[])
            ax.set_title(f'Label: {labels[i]}')
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))
        break
    


def print_data_stats(data_loader):
    classes = data_loader.dataset.dataset.classes
    class_count = {}
    for _, labels in data_loader:
      for label in labels:
        label = classes[label]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    print(class_count)
    

def print_train_log(epochs, train_acc, test_acc, train_loss, test_loss, learning_rates):
    print("\nEpoch\t     Train Loss\t     Test Loss     Train Accuracy    Test Accuracy    Learning Rate")
    print("===========================================================================================")
    for cnt in range(epochs):
        print(f"{cnt+1}\t\t{train_loss[cnt]:0.2f}\t\t{test_loss[cnt]:0.2f}\t\t{train_acc[cnt]:0.4f}\t\t{test_acc[cnt]:0.4f}\t\t{learning_rates[cnt]:0.8f}\n")

    print("===========================================================================================")
    


def plot_grad_cam_images(model, test_loader, device, class_names, rows=3, columns=5, transparency = 0.725):
    incorrect_examples, incorrect_pred,  incorrect_labels = get_incorrect_predictions(model, test_loader, device)
    fig,ax = plt.subplots(rows,columns)
    ax = ax.ravel()
    for i in range(rows*columns):
        image = incorrect_examples[0][i]
        t = torch.from_numpy(image)
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.24703223, 1/0.24348513, 1/0.26158784 ]),
                                    transforms.Normalize(mean = [ -0.49139968, -0.48215841, -0.44653091 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
        inv_tensor = invTrans(t)
        rgb_img = inv_tensor.permute(1,2,0)

        input_tensor = torch.tensor(inv_tensor.unsqueeze(0))
        target_layers = [model.layer3[-1]]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img.numpy(), grayscale_cam, use_rgb=True, image_weight=transparency)
        ax[i].imshow(visualization)
        ax[i].set_title(f"{class_names[incorrect_pred[0][i]]}/{class_names[incorrect_labels[0][i]]}")
        ax[i].set( xticks=[], yticks=[])
    plt.axis('off')
    plt.tight_layout()
    plt.xlabel('Misclassified Images')
    plt.ylabel('Loss')
    plt.legend("Incorrect Prediction/Actual Label")
    plt.show()
    
def model_summary(model, input_size):
    torchinfo.summary(model, 
                      input_size = input_size, 
                      batch_dim=0, 
                      col_names=("kernel_size",
                                 "input_size",
                                 "output_size",
                                 "num_params",
                                 "mult_adds"),
                       verbose=1,) 