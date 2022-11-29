import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import torch.cuda


# https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa

# Return: Array of length 5. Each element corresponds to the
# feature representation of each intermediate layer.
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Eliminate the unused layers(layers beyond conv5_1)
        self.req_features = ['0', '5', '10', '19', '28']  # Indices of layers 1,2,3,4,5
        # Drop all the rest layers from the features of the model,model will contain the first 29 layers
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        # initialize an array that will hold the activations from the chosen layers
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)

        return features


def image_loader(path, device):
    image = Image.open(path)
    # Preprocessing
    loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    # Add an extra dimension at 0th index for batch size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer by calculating the MSE between
    # the content and generated features and adding it to content loss
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l


def calc_style_loss(gen, style):
    # Calculating the gram matrix for the style and the generated image
    batch_size, channel, height, width = gen.shape

    G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

    # Calculating the style loss of each layer by calculating the MSE between
    # the gram matrix of the style image and the generated image and adding it to style loss
    style_l = torch.mean((G - A) ** 2)
    return style_l


def calculate_loss(generated_features, content_features, style_features, alpha, beta):
    style_loss = content_loss = 0
    for gen, cont, style in zip(generated_features, content_features, style_features):
        # extracting the dimensions from the generated image
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)

    total_loss = alpha * content_loss + beta * style_loss
    return total_loss

def perform_style_transfer(content, style):
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
    model = VGG().to(device).eval()

    content_image = image_loader(content, device)
    style_image = image_loader(style, device)
    generated_image = content_image.clone().requires_grad_(True)

    # Hyperparameters
    epoch = 7000
    lr = 0.004
    alpha = 8  # weighting coefficient of content loss
    beta = 70  # weighting coefficient of style loss

    # Updates the pixels of the generated image not the model parameter
    optimizer = optim.Adam([generated_image], lr=lr)

    # iterating for 1000 times
    for e in range(epoch):
        # extracting the features of generated, content and the original required for calculating the loss
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

        if e / 100:
            print(total_loss)
            save_image(generated_image, "generated.png")

# content_img = 'content.jpg'  # content: aggregated image from GANs
# style_img = 'style.jpg'  # base MRI image
# perform_style_transfer(content_img, style_img)