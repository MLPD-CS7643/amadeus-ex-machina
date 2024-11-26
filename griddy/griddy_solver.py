import time
import copy
import torch


class GriddySolver(object):
    def __init__(self, **kwargs):

        self.print = kwargs.pop("print", True)
        self.dtype = kwargs.pop("dtype", "float32")
        self.warmup = kwargs.pop("warmup", 0)
        self.steps = kwargs.pop("steps", [6, 8])
        self.epochs = kwargs.pop("epochs", 10)
        self.early_stop = kwargs.pop("early_stop", False)
        self.early_stop_epochs = kwargs.pop("early_stop_epochs", 10)
        self.batch_size = kwargs.pop("batch_size", 128)
        
        self.lr = kwargs.get("lr", 0.001)
        self.device = kwargs.get("device", "cuda")
        
        self.train_data = kwargs.pop("train_data") # DataLoader
        self.test_data = kwargs.pop("test_data") # DataLoader

        #TODO: add scheduler support

        #TODO: ensure no kwargs overlap
        optim_class = kwargs.pop("optim_class")
        self.optimizer = optim_class(**kwargs)

        loss_class = kwargs.pop("loss_class")
        self.loss = loss_class(**kwargs)

        model_class = kwargs.pop("model_class")
        self.model = model_class(**kwargs).to(self.device)

        if self.dtype == 'float16':
            self.model = self.model.half()
        elif self.dtype == 'bfloat16':
            self.model = self.model.bfloat16()

        if self.print:
            print(self.model)

        self._reset() #TODO: check if necessary

    def _reset(self):
        self.best = 0.0
        self.best_cm = None
        self.best_model = None

    def train(self):
        no_improve = 0
        best_loss = float('inf')
        stop_epoch = 0
        train_acc = []
        val_acc = []
        for epoch in range(self.epochs):
            stop_epoch = epoch
            print(f"EPOCH: {epoch}")
            self._adjust_learning_rate(epoch)

            # train loop
            t_acc = self._train_step(epoch)
            train_acc.append(t_acc.item())
            # validation loop
            acc, cm, loss = self._evaluate(epoch)
            val_acc.append(acc.item())
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if acc > self.best:
                self.best = acc
                self.best_cm = cm
                self.best_model = copy.deepcopy(self.model)
            
            if self.early_stop and no_improve >= self.early_stop_epochs:
                print("EARLY STOP E:{} L:{:.4f}".format(epoch, best_loss))
                break

        per_cls_acc = self.best_cm.diag().detach().numpy().tolist()
        if self.print:
            print("Best Prec @1 Acccuracy: {:.4f}".format(self.best))
            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))
        
        return self.best.item(), self.best_model, stop_epoch, train_acc, val_acc, per_cls_acc

    def _train_step(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        self.model.train()

        for idx, (data, target) in enumerate(self.train_data):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if self.print and idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.train_data),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        return acc.avg

    def _evaluate(self, epoch):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        num_class = 10
        cm = torch.zeros(num_class, num_class)
        self.model.eval()

        # evaluation loop
        for idx, (data, target) in enumerate(self.test_data):
            start = time.time()

            data = data.to(self.device)
            target = target.to(self.device)

            out, loss = self._compute_loss_update_params(data, target)

            batch_acc = self._check_accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss.item(), out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if self.print and idx % 10 == 0:
                print(
                    (
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t"
                    ).format(
                        epoch,
                        idx,
                        len(self.test_data),
                        iter_time=iter_time,
                        loss=losses,
                        top1=acc,
                    )
                )
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        if self.print:
            for i, acc_i in enumerate(per_cls_acc):
                print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm, losses.avg

    def _check_accuracy(self, output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        acc = correct / batch_size

        return acc

    def _compute_loss_update_params(self, data, target):
        output = None
        loss = None
        if self.model.training:
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                output = self.model(data)
                loss = self.loss(output, target)
        return output, loss

    def _adjust_learning_rate(self, epoch):
        epoch += 1
        if epoch <= self.warmup:
            lr = self.lr * epoch / self.warmup
        elif epoch > self.steps[2]:
            lr = self.lr * 0.2
        elif epoch > self.steps[1]:
            lr = self.lr * 0.3
        elif epoch > self.steps[0]:
            lr = self.lr * 0.5
        else:
            lr = self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count