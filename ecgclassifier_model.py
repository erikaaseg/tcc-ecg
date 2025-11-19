from typing import Any
import lightning as L
from torchvision.models import resnet34, ResNet34_Weights
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity
import torch


class ECGClassifier(L.LightningModule):
    def __init__(self,
                 num_classes: int = 4,
                 learning_rate: float = 1e-4):
        super(ECGClassifier, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self._load_backbone()

        self.criterion = torch.nn.CrossEntropyLoss()

        # Precisa ser um para cada conjunto (train, val, test) devido ao estado interno das métricas
        self.train_metrics = self._create_metrics()
        self.val_metrics = self._create_metrics()
        self.test_metrics = self._create_metrics()

    def _load_backbone(self):
        ### ResNet Backbone ###
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.num_classes)

    def _create_metrics(self):
        metrics = torch.nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass', num_classes=self.num_classes),
            'precision': Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall': Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1score': F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'specificity': Specificity(task='multiclass', num_classes=self.num_classes, average='macro')
        })
        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _common_step(self,
                     batch: tuple[torch.Tensor, torch.Tensor],
                     stage: str):
        images, labels = batch
        # images.shape: (Batch, Channels, Height, Width)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        metrics = getattr(self, f'{stage}_metrics')
        preds = torch.argmax(outputs, dim=1)
        for name, metric in metrics.items():
            metric.update(preds, labels)

        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True)

        return loss

    def training_step(self,
                      batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx):
        loss = self._common_step(batch, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, stage='val')

    def test_step(self, batch, batch_idx):
        self._common_step(batch, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _on_epoch_end_common(self, stage: str):
        metrics = getattr(self, f'{stage}_metrics')
        for name, metric in metrics.items():
            self.log(f'{stage}/{name}', metric.compute(), prog_bar=stage == 'val')
            metric.reset()

    def on_train_epoch_end(self):
        self._on_epoch_end_common(stage='train')

    def on_validation_epoch_end(self):
        self._on_epoch_end_common(stage='val')

    def on_test_epoch_end(self):
        self._on_epoch_end_common(stage='test')

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        batch = args[0]
        images, _ = batch
        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        return preds