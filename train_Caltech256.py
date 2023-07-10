import lightning.pytorch as pl
from torch import nn, optim, utils
from torchvision.datasets import Caltech256
from torchvision.transforms import ToTensor
from pyramid_vit.model import PyramidVisionTransformer

pyramid_vit = PyramidVisionTransformer(
    input_dims=(3, 256, 256),
    out_channels=[32, 64, 128, 256],
    patch_sizes=[4, 2, 2, 2],
    encoders_per_stage=[2, 2, 2, 2],
    reduction_ratio=[8, 4, 2, 1],
    num_heads=[1, 2, 5, 8],
    mlp_ratio=[8, 8, 4, 4]
)

class LitPyramidViTClassifier(pl.LightningModule):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.cross_entropy(x_hat, y, label_smoothing=0.1)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
