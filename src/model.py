# If not pretrained, initialize weights to small random values ("speeds up training")
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.models import (resnet50, ResNet50_Weights, efficientnet_v2_l,
                                EfficientNet_V2_L_Weights, squeezenet1_1,
                                SqueezeNet1_1_Weights, mobilenet_v2,
                                MobileNet_V2_Weights, vit_b_16,
                                ViT_B_16_Weights)

from utils.utils import mean, std


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data)
        # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data)
        # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# Only training final layer if feature_extracting=True
def set_parameter_requires_grad(model, freeze_backbone):
    if freeze_backbone:
        # print(model.parameters())
        for param in model.parameters():
            param.requires_grad = False


class MetaNet(nn.Module):

    def __init__(self, backbone, bb_output_size, num_classes):
        super(MetaNet, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(bb_output_size + 12,
                             64)  # 12 values from metadata
        self.fc2 = nn.Linear(64, num_classes)

        modules = [
            self.fc1, self.fc2
        ]  #TODO: find way to look through only layers added by this module
        for module in modules:
            weights_init(module)

    def forward(self, image, data):
        x1 = F.relu(self.backbone(image))
        x2 = data

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def update_final_layer(model, num_classes):
    name = model.__class__.__name__
    if name == "ResNet":
        in_features = model.fc.in_features
        module = nn.Linear(in_features, num_classes)
        module.apply(weights_init)
        model.fc = module
    elif name == "SqueezeNet":
        in_channels = model.classifier[1].in_channels
        module = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels,
                      num_classes,
                      kernel_size=(1, 1),
                      stride=(1, 1)), nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        module.apply(weights_init)
        model.classifier = module
    elif name == "EfficientNet":
        in_features = model.classifier[1].in_features
        module = nn.Linear(in_features, num_classes)
        module.apply(weights_init)
        model.classifier[1] = module
    elif name == "MobileNetV2":
        in_features = model.classifier[1].in_features
        module = nn.Linear(in_features, num_classes)
        module.apply(weights_init)
        model.classifier[1] = module
    elif name == "VisionTransformer":
        in_features = model.heads.head.in_features
        module = nn.Linear(in_features, num_classes)
        module.apply(weights_init)
        model.heads.head = module
    else:
        print(f"Invalid model name: {name}, exiting...")

    return model


def initialize_model(model_func,
                     num_classes,
                     use_metadata=False,
                     freeze_backbone=False,
                     pretrained_weights=None):
    assert (
        freeze_backbone != True or pretrained_weights is not None
    ), "You should not freeze the backbone unless you are using pretrained weights"

    model = model_func(weights=pretrained_weights)
    if pretrained_weights is None:
        model.apply(weights_init)
    set_parameter_requires_grad(model, freeze_backbone)
    if use_metadata:
        model = MetaNet(
            model, 1000, num_classes
        )  # remove hardcoding --> get layer to remove if not metadata and check out_channels
    else:
        model = update_final_layer(model, num_classes)

    return model


model_dicts = {
    "resnet": {
        "model": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2
    },
    "efficient": {
        "model": efficientnet_v2_l,
        "weights": EfficientNet_V2_L_Weights.IMAGENET1K_V1
    },
    "squeezenet": {
        "model": squeezenet1_1,
        "weights": SqueezeNet1_1_Weights.IMAGENET1K_V1
    },
    "convnext": {
        "model": mobilenet_v2,
        "weights": MobileNet_V2_Weights.IMAGENET1K_V2
    },
    "vit": {
        "model": vit_b_16,
        "weights": ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    }
}

def get_model(cf):
    model_dict = model_dicts[cf["model"]["name"]]
    num_classes = 1
    pretrained_weights = model_dict["weights"] if cf["model"]["pretrained_weights"] else None
    model = initialize_model(
        model_dict["model"],
        num_classes,
        use_metadata=cf["model"]["use_metadata"],
        freeze_backbone=cf["model"]["freeze_backbone"],
        pretrained_weights=pretrained_weights)
    return model


def get_base_transforms(cf):
    if cf["model"]["pretrained_weights"]:
        model_dict = model_dicts[cf["model"]["name"]]
        pretrained_weights = model_dict["weights"]
        transform = pretrained_weights.transforms()
    else:
        img_size = [cf["data"]["augs"]["resize"]]
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform


def main():
    for md in model_dicts.keys():
        weights = model_dicts[md]["weights"]
        print(md)
        print(weights.transforms())
        breakpoint()


if __name__ == "__main__":
    main()
