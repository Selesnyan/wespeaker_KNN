# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torchnet as tnt

from torch.nn.functional import sigmoid, softmax

def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              iter_per_epoch,
              logger,
              scaler,
              enable_amp,
              rank,
              trigger_sync,
              wandb=None,
              use_per_frame_embeddings=False,
              use_lennorm_embeddings=False,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    acc_correct = 0
    acc_nsamples = 0

    print('UPFE', use_per_frame_embeddings)
    
    # ~~8.26 %  of silence (in discretized windows)
    criterion_sos = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0/92.0))


    for i, batch in enumerate(dataloader):
        # utts = batch['key']
        targets = batch['label']
        features = batch['feat']

        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        features = features.float().to(device)  # (B,T,F)
        targets = targets.long().to(device)
        
        # print(use_lennorm_embeddings, use_per_frame_embeddings)
        
        with torch.cuda.amp.autocast(enabled=enable_amp):
            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            
            if use_lennorm_embeddings:
                embeds /= torch.unsqueeze(torch.norm(embeds, dim=1), dim=1)

            outputs = model.module.projection(embeds, None)

            if use_per_frame_embeddings:
                targets = targets.flatten() 
                loss = criterion(outputs, targets)


        # loss, acc
        loss_meter.add(loss.item())
        # acc_meter.add(out_sid.cpu().detach().numpy(), targets.cpu().numpy())
	
        acc_correct += torch.sum(torch.argmax(outputs.detach(), dim=1) == targets).item()
        acc_nsamples += outputs.shape[0]
    

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % log_batch_interval == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_correct / acc_nsamples),
                       width=10,
                       style='grid'))
            if wandb is not None and rank == 0:
                wandb.log({
                    'train/epoch': epoch,
                    'train/step': (epoch - 1) * iter_per_epoch + i,
                    'train/step/loss': loss_meter.value()[0],
                    'train/step/acc': acc_correct / acc_nsamples,
                    'train/step/lr': scheduler.get_lr(),
                })

                if trigger_sync is not None:
                    trigger_sync()

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row((epoch, i + 1, scheduler.get_lr(),
                margin_scheduler.get_margin()) +
               (loss_meter.value()[0], acc_correct / acc_nsamples),
               width=10,
               style='grid'))

    if wandb is not None and rank == 0:
        wandb.log({
            'train/ep/loss': loss_meter.value()[0],
            'train/ep/acc': acc_correct / acc_nsamples,
            'train/ep/lr': scheduler.get_lr(),
        })
        if trigger_sync is not None:
            trigger_sync()

