import os
import uuid
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.trainer import Trainer
from recbole.utils import set_color, get_gpu_usage


class DAGCLTrainer(Trainer):
    """Trainer for DAGCL — adds unique saved-model filename to avoid collisions."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._make_saved_model_file_unique()

    def _make_saved_model_file_unique(self):
        saved_model_file = getattr(self, "saved_model_file", None)
        if not saved_model_file:
            return
        root, ext = os.path.splitext(saved_model_file)
        run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.saved_model_file = f"{root}-{run_id}{ext}"

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(train_data, total=len(train_data), ncols=100,
                 desc=set_color(f"Train {epoch_idx:>5}", 'pink'))
            if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss
