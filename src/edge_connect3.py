import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset3 import Dataset
from .model3 import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
import subprocess

class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        # if self.config.MODE == 2:
        #     # self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_INPAINTING_FLIST,config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=True)
        #     self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_INPAINTING_FLIST,config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=True)
        #
        # else:
        self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_INPAINTING_FLIST,config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=False, training=True)
        self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_INPAINTING_FLIST,config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
        self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_INPAINTING_FLIST,config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=True)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()
            self.inpaint_model.save2()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        value_epoch=0
        max_value_iteration=0
        max_precision, max_recall=0,0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        epoch_precision = 0.0
        epoch_recall = 0.0
        total_batches = len(train_loader)

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            # 重置累积值
            epoch_precision = 0.0
            epoch_recall = 0.0

            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_gray, edges, masks,images_inpaint, masks_noi = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks,masks_noi)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))

                    # # 更新epoch的precision和recall累积值
                    # epoch_precision += precision.item()
                    # epoch_recall += recall.item()
                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks,images_inpaint)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks,masks_noi)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks,images_inpaint)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks,images_inpaint)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration



                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                # if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                #     self.sample()
                    # self.test()

                # evaluate model at checkpoints
                # if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                #     print('\nstart eval...\n')
                #     value_epoch+=1
                #     print('Eval epoch:',value_epoch)
                #     # max_value_iteration,max_precision, max_recall=self.eval(value_epoch,max_value_iteration,max_precision, max_recall)
                #     self.eval()
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
            # 计算平均precision和recall
            # epoch_precision /= total_batches
            # epoch_recall /= total_batches

            # # 打印平均precision和recall
            # print('\nEpoch %d, Average Precision: %.4f, Average Recall: %.4f' % (
            #     epoch, epoch_precision, epoch_recall))
            # if epoch_precision > 0.95 and epoch_recall > 0.95:
            #     self.save()
            #     keep_training = False
            #     break

        print('\nEnd training....')
    # def eval2(self):
    #     self.test()

    # def eval(self,value_epoch,max_value_iteration,max_precision, max_recall):
    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=False,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        epoch_precision_val = 0.0
        epoch_recall_val = 0.0
        total_batches_val = len(val_loader)

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks,images_inpaint , masks_noi = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks,masks_noi)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                # 更新epoch的precision和recall累积值
                epoch_precision_val += precision.item()
                epoch_recall_val += recall.item()

            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images ,edges, masks,images_inpaint)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                # self.test()

            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks,masks_noi)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks,images_inpaint)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks,images_inpaint)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs


            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

        if model == 1:
            # 计算平均precision和recall
            epoch_precision_val /= total_batches_val
            epoch_recall_val /= total_batches_val


            # # 创建用于存储 precision 和 recall 的列表
            # precision_values = [item[1] for item in logs if item[0] == 'precision']
            # recall_values = [item[1] for item in logs if item[0] == 'recall']

            # # 找到 precision 和 recall 的最大值
            # max_precision = max(precision_values)
            # max_recall = max(recall_values)

            # if epoch_precision_val>max_precision and epoch_recall_val>max_recall:
            #     max_value_iteration=value_epoch
            #     max_precision, max_recall=epoch_precision_val,epoch_recall_val
            #     self.save()
            # print('precision: {:.4f}, recall: {:.4f}'.format(epoch_precision_val, epoch_recall_val))
            # print('iteration: {},max_precision: {:.4f}, max_recall: {:.4f}'.format(max_value_iteration, max_precision,
            #                                                                        max_recall))
            # return max_value_iteration,max_precision, max_recall

    # def test2(self):
    #     self.test()

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL

        create_dir(self.results_path)
        print(model)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )


        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks ,images_inpaint, masks_noi = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks,masks_noi)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))
            # elif model == 4:
            #     outputs = self.edge_model(images_gray, edges, masks)
            #     outputs_merged = outputs * masks

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images ,edges, masks, images_inpaint)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                # outputs_merged=edges
            # elif model == 4:
            #     # outputs_merged= masks_noi
            #     outputs_merged=  images * (1 - masks)+masks

            # elif model == 4:
            #     # outputs = self.inpaint_model(images, edges, masks, images_inpaint)
            #     # outputs_merged = (outputs * masks) + (images * (1 - masks))
            #     # outputs_merged = outputs
            #     outputs_merged=edges* (1 - masks)

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks,masks_noi).detach()
                outputs = self.inpaint_model(images, edges, masks,images_inpaint)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')
        subprocess.run(["python", "/home/user/egde-connect-self/scripts/metrics.py",  "--data",os.path.dirname(self.config.TEST_FLIST),"--output", self.results_path])


    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks, images_inpaint, masks_noi= self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks_noi
            outputs = self.edge_model(images_gray, edges, masks, masks_noi)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            # inputs = (images * (1 - masks)) + masks
            inputs=images_inpaint
            outputs = self.inpaint_model(images, edges, masks, images_inpaint)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            # inputs = (images * (1 - masks)) + masks
            inputs=images_inpaint
            outputs = self.edge_model(images_gray, edges, masks,masks_noi).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks,images_inpaint)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
