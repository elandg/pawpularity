import torch
import torch.optim as optim
import pytorch_lightning as pl

from utils.model import get_model
from utils.utils import metrics

class PawpularitySystem(pl.LightningModule):

    def __init__(self, cf):
        super().__init__()
        self.cf = cf
        use_metadata = cf["model"]["use_metadata"]
        if use_metadata:
            self.forward_func = self.forward_metadata_
        else:
            self.forward_func_ = self.forward_no_metadata_

        self.model = get_model(cf)

        self.train_loss = metrics[cf["model"]["loss"]]
        self.val_loss = metrics[cf["model"]["loss"]]
        # self.metrics = torch.nn.ModuleDict(
        #     {k: metrics[k]
        #      for k in cf["model"]["metrics"]})

    def forward_no_metadata_(self, image, metadata):
        return self.model(image)

    def forward_metadata_(self, image, metadata):
        return self.model(image, metadata)

    def forward(self, image, metadata):
        return self.forward_func_(image, metadata)

    def training_step(self, batch, batch_idx):
        image, metadata, score = batch["image"], batch["metadata"], batch[
            "score"].unsqueeze(1)
        output = self(image, metadata)
        loss = self.train_loss(output, score)

        bs = image.shape[0]
        self.log("train_loss", loss, on_step=True, batch_size=bs)

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        image, metadata, score = batch["image"], batch["metadata"], batch[
            "score"].unsqueeze(1)
        output = self(image, metadata)

        self.val_loss.update(output, score)

        # for metric in self.metrics.keys():
        #     self.metrics[metric].update(output, score)

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log(f"val/{self.cf['model']['loss']}", val_loss, on_epoch=True)
        self.val_loss.reset()

        # # compute metrics
        # for metric in self.metrics.keys():
        #     self.log(f"val_{metric}",
        #              self.metrics[metric].compute(),
        #              on_epoch=True)

        #     # reset metrics
        #     self.metrics[metric].reset()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        image, metadata, score = batch["image"], batch["metadata"], batch[
            "score"].unsqueeze(1)
        output = self(image, metadata)
        return output, score

    def configure_optimizers(self):
        # Passing frozen layers into optimizer should produce an error
        # "requires_grad has to be false AND the parameter cannot be given to the optimizer" -https://www.reddit.com/r/MLQuestions/comments/t3ipan/pytorchlightning_trainerfit_method_is_unfreezing/
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8, dampening=0, weight_decay=0.00005)
        return optim.Adam(filter(lambda p: p.requires_grad,
                                 self.model.parameters()),
                          lr=self.cf["model"]["lr"])