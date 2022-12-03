import copy

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.utils import save_image


# Reference: https://github.com/Aleadinglight/Pytorch-VGG-19/blob/master/VGG_19.ipynb
class ContentLoss(nn.Module):
    def __init__(self, base, ):
        super(ContentLoss, self).__init__()
        self.base = base.detach()

    def forward(self, img):
        self.loss = torch.nn.functional.mse_loss(img, self.base)
        return img


class StyleLoss(nn.Module):
    def __init__(self, base_feature):
        super(StyleLoss, self).__init__()
        self.base = calculate_gram_matrix(base_feature).detach()

    def forward(self, img):
        gram = calculate_gram_matrix(img)
        self.loss = torch.nn.functional.mse_loss(gram, self.base)
        return img


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def calculate_gram_matrix(img):
    batch_size, feature_maps_num, feature_map_dim1, feature_map_dim2 = img.size()
    features = img.view(batch_size * feature_maps_num, feature_map_dim1 * feature_map_dim2)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * feature_maps_num * feature_map_dim1 * feature_map_dim2)


def image_loader(path, device):
    image = Image.open(path)
    loader = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    # Add an extra dimension at 0th index for batch size
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def calculate_losses(cnn, normalization_mean, normalization_std, style_image, content_image, device):
    cnn = copy.deepcopy(cnn)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer:', layer.__class__.__name__)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def create_model():
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
    model = models.vgg19(pretrained=True).features.to(device).eval()

    return device, model

def perform_style_transfer(model, device, content_image, style_image, img_num,
                           epochs=300, style_weight=1000000, content_weight=5):

    content = image_loader(content_image, device)
    style = image_loader(style_image, device)
    generated = content.clone()

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model, style_losses, content_losses = calculate_losses(model, normalization_mean,
                                                           normalization_std, style, content, device)
    optimizer = optim.LBFGS([generated.requires_grad_()])

    epoch = [0]
    losses = []

    while epoch[0] <= epochs:
        def closure():
            generated.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(generated)
            style_score = 0
            content_score = 0

            for style_loss in style_losses:
                style_score += style_loss.loss

            for content_loss in content_losses:
                content_score += content_loss.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 100 == 0:
                losses.append(loss.item())
                print('\nFor epoch', epoch[0], 'Style Loss:', style_score.item(),
                      'Content Loss:', content_score.item())

                file_name = str(img_num) + ' - epoch ' + str(epochs) + ' a ' + str(content_weight) + ' b ' + str(style_weight)
                save_image(generated, 'generatedimageslbfgs/' + file_name + '.png')

            return style_score + content_score

        optimizer.step(closure)

    generated.data.clamp_(0, 1)

    plt.rcParams["figure.figsize"] = [8, 4]

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(range(1, epochs + 1, 100), losses)
    plt.show()

    return generated


epochs_num = [500]
content_weights = [5]
style_weights = [1000, 100]
samples_num = 10

for e in epochs_num:
    for a in content_weights:
        for b in style_weights:
            for img_num in range(1, samples_num + 1):
                content_img_filename = 'wgan_gp/t1/fake/' + str(img_num) + '.png'
                style_img_filename = 'wgan_gp/t1/real/' + str(img_num) + '.png'

                device, model = create_model()

                perform_style_transfer(model, device, content_img_filename, style_img_filename,
                                                   img_num,
                                                   epochs=e,
                                                   style_weight=b, content_weight=a)
