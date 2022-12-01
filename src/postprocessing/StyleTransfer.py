import matplotlib.pyplot as plt
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib.ticker import MaxNLocator
from torchvision.utils import save_image


# Reference: https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa

# Return: Array of length 5. Each element corresponds to the
# feature representation of each intermediate layer.
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Eliminate the unused layers(layers beyond conv5_1)
        self.layers = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for index, layer in enumerate(self.model):
            x = layer(x)
            if str(index) in self.layers:
                features.append(x)

        return features


def calculate_content_loss(generated_features, content_features):
    content_loss = torch.mean((generated_features - content_features) ** 2)
    return content_loss


def calculate_style_loss(generated, style):
    batch_size, channel, height, width = generated.shape

    G = torch.mm(generated.view(channel, height * width), generated.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

    style_l = torch.mean((G - A) ** 2)
    return style_l


def calculate_loss(generated_features, content_features, style_features, alpha, beta):
    style_loss = content_loss = 0
    for generated, content, style in zip(generated_features, content_features, style_features):
        content_loss += calculate_content_loss(generated, content)
        style_loss += calculate_style_loss(generated, style)

    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


def image_loader(path, device):
    image = Image.open(path)

    loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    # Add an extra dimension at 0th index for batch size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def perform_style_transfer(content, style):
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
    model = VGG().to(device).eval()

    content_image = image_loader(content, device)
    style_image = image_loader(style, device)
    generated_image = content_image.clone().requires_grad_(True)

    # Hyperparameters
    epochs = 2000
    lr = 0.004
    alpha = 5  # weighting coefficient of content loss
    beta = 100  # weighting coefficient of style loss

    # Updates the pixels of the generated image not the model parameter
    optimizer = optim.Adam([generated_image], lr=lr)
    loss = []

    for epoch in range(1, epochs + 1):

        generated_features = model(generated_image)
        content_features = model(content_image)
        style_features = model(style_image)

        # iterating over the activation of each layer and calculate the loss and
        # add it to the content and the style loss
        total_loss = calculate_loss(generated_features, content_features, style_features, alpha, beta)

        # optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()  # Backpropagate the total loss
        optimizer.step()  # Update the pixel values of the generated image

        if epoch % 50 == 0:
            print("For", epoch, "epochs, total loss is:", str(total_loss.item()))
            loss.append(total_loss.item())
            save_image(generated_image, "generated_image_adam.png")

    plt.rcParams["figure.figsize"] = [8, 4]

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(range(1, epochs + 1, 50), loss)
    plt.show()


content_img = 'content.jpg'  # content: aggregated image from GANs
style_img = 'style.jpg'  # base MRI image
perform_style_transfer(content_img, style_img)
