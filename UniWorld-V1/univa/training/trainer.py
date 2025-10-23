from transformers import Trainer


class UniVATrainer(Trainer):
    def create_optimizer(self):
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = []
        trainable_params = []
        if self.optimizer is None:
            for n, p in self.model.named_parameters():
                if n in decay_parameters and p.requires_grad:
                    optimizer_grouped_parameters += [
                        {
                            "params": [p],
                            "weight_decay": self.args.weight_decay,
                        }
                    ]
                    trainable_params.append(n)
                elif n not in decay_parameters and p.requires_grad:
                    optimizer_grouped_parameters += [
                        {
                            "params": [p],
                            "weight_decay": 0.0,
                        }
                    ]
                    trainable_params.append(n)
                else:
                    p.requires_grad_(False)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        return self.optimizer
