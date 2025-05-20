import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# -----------------------------------------#
#   Filter UserWarning
# -----------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -----------------------------------------#
#   Set random seed
# -----------------------------------------#
SEEDS = 42
np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)
os.environ['PYTHONHASHSEED'] = str(SEEDS)

# -----------------------------------------#
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.98 #
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# -----------------------------------------#


from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

# Set true to show less logging messages
os.environ["WANDB_SILENT"] = "false"

# ----------------------------------------------------------#
from model.unet import ConvNeXt_Unet

# ----------------------------------------------------------#
from model.training import (CE,Focal_Loss, dice_loss_with_CE,
                                dice_loss_with_Focal_Loss, get_lr_scheduler)
from utils import DisplayCallback,EvalCallback, LossHistory, ParallelModelCheckpoint,Iou_score, f_score, show_config
from dataset.dataloader import UnetDataset
from configs import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import logger


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in config.train_gpu)
    ngpus_per_node                      = len(config.train_gpu)


    model = ConvNeXt_Unet(input_shape=(config.input_shape[0], config.input_shape[1], 3), num_classes=4, backbone=config.backbone)

    if config.model_path != '':
        #------------------------------------------------------
        #   load pretrained weights
        #------------------------------------------------------
        if True:
            model.load_weights(config.model_path, by_name=True, skip_mismatch=True)


    else:
        print("Weights initialized!")

    #--------------------------
    #   loss functions
    #--------------------------
    if config.focal_loss:
        if config.dice_loss:
            loss = dice_loss_with_Focal_Loss(config.cls_weights)
        else:
            loss = Focal_Loss(config.cls_weights)
    else:
        if config.dice_loss:
            loss = dice_loss_with_CE(config.cls_weights)
        else:
            loss = CE(config.cls_weights)

    #---------------------------
    #   load txt file
    #---------------------------
    with open(os.path.join(config.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(config.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
        num_classes = config.num_classes, backbone = config.backbone, model_path = config.model_path, input_shape = config.input_shape, \
        unlabeled_id_path = config.unlabeled_id_path, Freeze_Train = config.Freeze_Train, \
        Init_lr = config.Init_lr, Min_lr = config.Min_lr, lr_decay_type = config.lr_decay_type, \
        num_workers = config.num_workers, num_train = num_train, num_val = num_val
    )
    logger.info(f'>>>> Total params : {model.count_params() / 1e6 : .1f}M\n')
    # logger.info('>>>> Load pretrained weights from %s\n' % config.model_path)

    print("Starting Training Loop...")
    logger.info('-------------------------- start retraining Itr3 --------------------------')

    logger.info('===========> Total stage 1/2: Freeze training')
    if True:
        if config.Freeze_Train:
            #------------------------------------#
            #   Freeze backbone
            #------------------------------------#
            if config.backbone == "vgg":
                freeze_layers = 17
            elif config.backbone == "resnet50":
                freeze_layers = 141
            elif   config.backbone == "ConvNeXtTiny":
                freeze_layers = 148
            else:
                raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50, efficientnet.'.format(config.backbone))
            for i in range(freeze_layers):
                   model.layers[i].trainable = False

            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))


        # batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        batch_size = config.Freeze_batch_size
        start_epoch = config.Init_Epoch
        end_epoch   = config.Freeze_Epoch if config.Freeze_Train else config.UnFreeze_Epoch
        # end_epoch = config.Freeze_Epoch

        nbs             = 16
        lr_limit_max    = 1e-4 if config.optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if config.optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        from tensorflow.keras.optimizers import Adam

        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            decay=0.0,
            amsgrad=False
        )


        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = config.momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = config.momentum, nesterov=True),
           }[config.optimizer_type]

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[f_score()])

        lr_scheduler_func = get_lr_scheduler(config.lr_decay_type, Init_lr_fit, Min_lr_fit, config.UnFreeze_Epoch)

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The training dataset is too small, please enlarge your dataset!')

        #-------------------------------
        #   dataloader
        #-------------------------------
        train_dataloader    = UnetDataset(train_lines, config.input_shape, batch_size, config.num_classes,'train', config.VOCdevkit_path)
        val_dataloader      = UnetDataset(val_lines, config.input_shape, batch_size, config.num_classes, 'val', config.VOCdevkit_path)


        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(config.save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        if ngpus_per_node > 1:
            checkpoint      = ParallelModelCheckpoint(model, os.path.join(config.save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = config.save_period)
            checkpoint_last = ParallelModelCheckpoint(model, os.path.join(config.save_dir, "last_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ParallelModelCheckpoint(model, os.path.join(config.save_dir, "best_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        else:
            checkpoint      = ModelCheckpoint(os.path.join(config.save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = config.save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(config.save_dir, "last_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(config.save_dir, "best_epoch_weights.h5"),
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        eval_callback   = EvalCallback(model, config.input_shape, config.num_classes, val_lines, config.VOCdevkit_path, log_dir, \
                                        eval_flag=config.eval_flag, period=config.eval_period)
        display_callback = DisplayCallback(model, config.input_shape, val_lines, config.VOCdevkit_path, epoch_interval=1)
        callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            freeze_history = model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if config.num_workers > 1 else False,
                workers             = config.num_workers,
                callbacks           = callbacks
            )


        if config.Freeze_Train:
            logger.info('------------------------- finetune training -------------------------')
            logger.info('===========> Total stage 2/2: Finetune training')
            batch_size  = config.Unfreeze_batch_size
            start_epoch = config.Freeze_Epoch if start_epoch < config.Freeze_Epoch else start_epoch
            end_epoch   = config.UnFreeze_Epoch

            nbs             = 16
            lr_limit_max    = 1e-4 if config.optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if config.optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(config.lr_decay_type, Init_lr_fit, Min_lr_fit, config.UnFreeze_Epoch)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)

            callbacks = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler,
                         eval_callback]


            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(loss = loss,
                          optimizer = optimizer,
                          metrics = [f_score()])

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError('The training dataset is too small, please enlarge your dataset!')

            train_dataloader.batch_size    = config.Unfreeze_batch_size
            val_dataloader.batch_size      = config.Unfreeze_batch_size

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            finetune_history = model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if config.num_workers > 1 else False,
                workers             = config.num_workers,
                callbacks           = callbacks,
           )


            logger.info('------------------------- training finished -------------------------')