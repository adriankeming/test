import base64
import io
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings

from .forms import ImageUploadForm
###
#相對路徑搜尋
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
#Path of parent directory
parentDirectory = os.path.dirname(fileDirectory)
#Navigate to Strings directory
PATH = os.path.join(parentDirectory, '旻家model_resnet18.pth' )   


train_model_name='旻家' #佳欣 旻家 明賢 榮昇 珮如 仲容
# PATH='旻家model_resnet18' 
# PATH = 'C:\\Users\\user\\desktop\\handwriting_recognition_project\\旻家model_resnet18.pth'
classes = ('false', 'true') # 分類的類別
num_classes = len(classes)
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(8192, num_classes) # 原512改8192

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
  return ResNet(ResidualBlock, num_classes=num_classes)

###
model = ResNet18()
model.load_state_dict(torch.load(PATH,map_location='cpu'))
model.eval()
###
# img = cv2.imread(f'{path_c}/{i}')
# my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((128,128)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# img_tensor = my_transforms(img)
# test1 = img_tensor.unsqueeze(0)
# outputs = model(test1)
# _, predicted = torch.max(outputs.data, 1)
###
# json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
# imagenet_mapping = json.load(open(json_path))


def transform_image(image_bytes):

    my_transforms = transforms.Compose([transforms.Resize((128,128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.5, 0.5, 0.5],
                                            [0.5, 0.5, 0.5])])
    img = Image.open(io.BytesIO(image_bytes))
    return my_transforms(img).unsqueeze(0)
    # image = Image.open(io.BytesIO(image_bytes))
    # return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    # tensor = transform_image(image_bytes)
    # outputs = model.forward(tensor)
    # _, y_hat = outputs.max(1)
    # predicted_idx = str(y_hat.item())
    # class_name, human_label = imagenet_mapping[predicted_idx]
    img_tensor = transform_image(image_bytes)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)
    if str(classes[predicted[0]])=='true':
        ans='本人'
    else:
        ans='非本人'
    return ans


def index(request):
    image_uri = None
    predicted_label = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # passing the image as base64 string to avoid storing it to DB or filesystem
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            # get predicted label
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'index_hand.html', context)
    # return render(request, 'Handwriting_recognition/index_hand.html', context)