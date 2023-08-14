import os
from pathlib import Path
from typing import Tuple


class ModelTrainer:
    def __init__(self, cfg: dict, yolo_engine: str):
        self.cfg = cfg
        self.yolo_engine = yolo_engine

    def _get_train_params(self) -> Tuple[str, bool, str]:
        required_params = (
            f'--data {Path(self.cfg["dataset"]["yolo_dataset"], "dataset.yaml")} '
            f'--imgsz {self.cfg["model"]["imgsz"]} '
            f'--epochs {self.cfg["model"]["epochs"]} '
            f'--batch-size {self.cfg["model"]["batch_size"]} '
            f'--cfg {self.cfg["model"]["model"]}.yaml '
            f'--weights {self.cfg["model"]["weights"]} '
            f'--hyp {self.cfg["model"]["model_params"]} '
            f'--project {self.cfg["model"]["project"]} '
            f'--name {self.cfg["model"]["run_name"]} '
            f'--patience {self.cfg["model"]["patience"]} '
            f'{"--cache"*self.cfg["model"]["cache"]} '
        )

        if self.cfg["model"]["ddp"]["is_on"]:
            ddp_params = (
                f'{"--sync-bn"*self.cfg["model"]["ddp"]["sync_bn"]} '
                f'--device {self.cfg["model"]["ddp"]["device"]}'
            )
            required_params += ddp_params


        return required_params, self.cfg["model"]["ddp"]["is_on"], self.cfg["model"]["ddp"]["gpu_n"]

    def train(self):
        # w&b off
        os.system(f'cd {self.yolo_engine} && wandb disabled')

        train_params, ddp_is_on, gpu_n = self._get_train_params()
        run_params = f"-m torch.distributed.run --nproc_per_node {gpu_n}" if ddp_is_on else ""

        os.system(f"cd {self.yolo_engine} && python {run_params} train.py {train_params}")
