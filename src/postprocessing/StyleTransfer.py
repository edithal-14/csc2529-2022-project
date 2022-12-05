import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
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

    gram = torch.mm(generated.view(channel, height * width), generated.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

    style_l = torch.mean((gram - A) ** 2)
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
    loader = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def create_model():
    device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')
    model = VGG().to(device).eval()

    return device, model


def perform_style_transfer(model, device, content_imgs, style_img, image_num, epochs=800, alpha=5, beta=100,
                           optimizer_betas=(0.9, 0.999)):
    style_image = image_loader(style_img, device)

    psnr_f = open("psnr.txt", "a")
    total_psnr = 0

    for img in content_imgs:
        content_image = image_loader(img, device)
        generated_image = content_image.clone().requires_grad_(True)

        # Updates the pixels of the generated image not the model parameter
        optimizer = optim.Adam([generated_image], betas=optimizer_betas)  # (0.5, 0.999)

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

        psnr_value = psnr(style_image.cpu().detach().numpy(), generated_image.cpu().detach().numpy())
        hyperparamters = str(epochs) + ',' + str(alpha) + ',' + str(beta)
        psnr_result = '\n' + str(psnr_value) + ',' + hyperparamters

        psnr_f.write(str(psnr_result))

        total_psnr += psnr_value

        # file_name = str(img_num) + ' - epoch ' + str(epochs) + ' a ' + str(alpha) + ' b ' + str(beta)
        file_name = str(img_num)

        save_image(generated_image, '../images/t1/wgan_gp_style/' + file_name + '.png')

    psnr_f.close()

    return total_psnr / len(content_imgs)


device, model = create_model()

epochs_num = [500]
alphas = [5]
betas = [100]
samples_num = 5

f = open("mean psnr.txt", "a")

for e in epochs_num:
    for a in alphas:
        for b in betas:
            psnr_val = 0
            for img_num in range(0, samples_num + 1):
                content_img_filename = ['../images/t1/wgan_gp/' + str(img_num) + '.png']
                style_img_filename = '../images/t1/real/' + str(img_num) + '.png'

                psnr_val += perform_style_transfer(model, device, content_img_filename, style_img_filename,
                                                   img_num,
                                                   epochs=e,
                                                   alpha=a, beta=b)

            mean_psnr = psnr_val / samples_num
            hyperparameters = str(e) + ',' + str(a) + ',' + str(b)
            result = '\n' + str(mean_psnr) + ',' + hyperparameters
            f.write(str(result))

            print("Mean PSNR values is", mean_psnr, "using: epoch =", str(e) + ', alpha =', str(a) + ', beta =', str(b))
