import os
import time

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    batch_train_time_list = []
    batch_val_time_list = []

    if local_rank == 0:
        print('\nStart Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    start_epoch_train_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value = loss_value_all

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                start_batch_train_time = time.time()

                outputs         = model_train(images)

                end_batch_train_time = time.time()
                batch_train_time = end_batch_train_time - start_batch_train_time
                batch_train_time_list.append(batch_train_time)

                loss_value_all  = 0
                #----------------------#
                #   计算损失
                #----------------------#
                for l in range(len(outputs)):
                    with torch.cuda.amp.autocast(enabled=False):
                        predication = outputs[l].float()
                    loss_item = yolo_loss(l, predication, targets)
                    loss_value_all  += loss_item
                loss_value = loss_value_all

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    end_epoch_train_time = time.time()
    epoch_train_time = end_epoch_train_time - start_epoch_train_time

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print("Train time per batch: {:.4f} ms | {:.4f} s".format((sum(batch_train_time_list) / len(batch_train_time_list)) * 1000, sum(batch_train_time_list) / len(batch_train_time_list)))
        print("Train time per epoch: {:.4f} ms | {:.4f} s".format(epoch_train_time * 1000, epoch_train_time))
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    start_val_time = time.time()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            start_batch_val_time = time.time()

            outputs         = model_train(images)

            end_batch_val_time = time.time()
            batch_val_time = end_batch_val_time - start_batch_val_time
            batch_val_time_list.append(batch_val_time)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    end_val_time = time.time()
    validation_time = end_val_time - start_val_time

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        print("Validation time per batch: {:.4f} ms | {:.4f} s".format((sum(batch_val_time_list) / len(batch_val_time_list)) * 1000, sum(batch_val_time_list) / len(batch_val_time_list)))
        print("Validation time per epoch: {:.4f} ms | {:.4f} s".format(validation_time * 1000, validation_time))
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

        checkpoint_dict = {'epoch': epoch + 1,
                           'model_state_dict':model.state_dict()}
        torch.save(checkpoint_dict, os.path.join(save_dir, "checkpoint.pth"))
        print("Checkpoint for epoch {} has been created: checkpoint.pth".format(epoch + 1))