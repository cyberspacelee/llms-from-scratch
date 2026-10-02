"""training 子包公开入口；具体公式与 Shape 见各实现模块。"""

from .losses import language_model_loss, next_token_loss, teacher_forcing, token_loss

__all__ = ["language_model_loss", "next_token_loss", "teacher_forcing", "token_loss"]
