import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers import ViTForImageClassification, ViTConfig
from loguru import logger
from utils.utils import *


class Arguments:

    def __init__(self):
        # for data
        self.data_type = "train"
        self.dataset_path = Path.cwd() / "data/datasets"

        # for huggingface and pretrained model_name_or_path
        self.model_name_or_path = Path.cwd() / "model_name_or_path"
        self.model_trained = Path.cwd() / "model_trained"   # the best model for next resume

        # for training
        self.training = True
        self.cuda = True
        self.distributed = False
        self.fp16 = False
        self.input_shape = [224, 224]
        self.resume = False
        self.load_from_pth = False
        self.batch_size = 8
        self.epoch = 3
        self.init_lr = 1e-2
        self.min_lr = self.init_lr * 0.01
        self.lr_decay_type = "cos"
        self.optimizer_type = "sgd"
        self.num_workers = 4

        self.optimizer_type = "sgd"
        self.momentum = 0.9
        self.weight_decay = 5e-4


class VitDataSet(torch.utils.data.Dataset):

    def __init__(self, args: Arguments, data_type: str = None):
        if not data_type or not args.dataset_path or not args.model_name_or_path:
            raise ValueError("voc dataset path or data type or pretrained model_name_or_path path is None")

        self.annotation_lines = load_cat_dog_dataset(str(args.dataset_path), data_type=data_type)
        self.image_processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path)
        logger.info(data_type + " datasets: " + str(len(self.annotation_lines)))

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        index = index % self.__len__()
        image = Image.open(self.annotation_lines[index][0])

        # get pixel_values
        inputs = self.image_processor(image, return_tensors="pt").pixel_values

        # get label
        label = self.annotation_lines[index][1]
        return inputs, label


class Trainer:

    def __init__(self,
                 args: Arguments = None,
                 train_dataset: VitDataSet = None,
                 test_dataset: VitDataSet = None):
        self.args = args

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # get id2label and label2id
        self.classes, self.id2label, self.label2id = ["cat", "dog"], {0: "cat", 1: "dog"}, {"cat": 0, "dog": 1}

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # loading model_name_or_path
        # 1
        self.model = ViTForImageClassification(config=ViTConfig())
        if not self.args.resume and self.args.model_name_or_path:
            logger.info(f"Loading model_name_or_path from: {self.args.model_name_or_path}")
            self.model = ViTForImageClassification.from_pretrained(args.model_name_or_path,
                                                                   id2label=self.id2label,
                                                                   label2id=self.label2id)

        # 2
        if self.args.resume and self.args.model_trained:
            logger.info(f"Loading pretrained model from: {self.args.model_trained} for resume training!")
            self.model = ViTForImageClassification.from_pretrained(self.args.model_trained,
                                                                   id2label=self.id2label,
                                                                   label2id=self.label2id)

        # 3
        if self.args.load_from_pth and self.args.model_trained:
            for f in Path(self.args.model_trained).iterdir():
                if f.suffix == ".pth":
                    logger.info(f"Loading pretrained model_name_or_path state dict from: {str(f)}")
                    model_dict = self.model.state_dict()
                    pretrained_dict = torch.load(f, map_location=self.device)
                    load_key, no_load_key, temp_dict = [], [], {}
                    for k, v in pretrained_dict.items():
                        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                            temp_dict[k] = v
                            load_key.append(k)
                        else:
                            no_load_key.append(k)
                    model_dict.update(temp_dict)
                    self.model.load_state_dict(model_dict)
                    break

        if self.args.fp16:
            logger.info("FP16")
            from torch.cuda.amp import GradScaler as GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.model_train = self.model.train().to(self.device)
        if self.args.cuda:
            self.model_train = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            self.model_train.cuda()

    def train(self):
        logger.info("Start training...")
        nbs = 64
        lr_limit_max = 1e-3 if self.args.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if self.args.optimizer_type == 'adam' else 5e-4
        init_lr_fit = min(max(self.args.batch_size / nbs * self.args.init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(self.args.batch_size / nbs * self.args.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type=self.args.lr_decay_type,
                                             lr=init_lr_fit,
                                             min_lr=min_lr_fit,
                                             total_iters=self.args.epoch)
        # set optimizer
        optimizer_dict = {
            'adam': torch.optim.Adam(self.model_train.parameters(),
                                     lr=init_lr_fit,
                                     betas=(self.args.momentum, 0.999),
                                     weight_decay=self.args.weight_decay),
            'sgd': torch.optim.SGD(self.model_train.parameters(),
                                   lr=init_lr_fit,
                                   momentum=self.args.momentum,
                                   nesterov=True)
        }
        optimizer = optimizer_dict[self.args.optimizer_type]

        loss_max = float("inf")
        for epoch in range(self.args.epoch):
            step_train = len(self.train_dataset) // self.args.batch_size
            step_test = len(self.test_dataset) // self.args.batch_size

            if step_train == 0 or step_test == 0:
                raise ValueError(f"step_train: {step_train} or step_test: {step_test} is too small")

            train_dataloader = DataLoader(self.train_dataset,
                                          shuffle=True,
                                          batch_size=self.args.batch_size,
                                          num_workers=self.args.num_workers,
                                          pin_memory=True,
                                          drop_last=True,
                                          collate_fn=collate_fn_cat_dog
                                          )
            test_dataloader = DataLoader(self.test_dataset,
                                         shuffle=False,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate_fn_cat_dog
                                         )
            set_optimizer_lr(optimizer=optimizer, lr_scheduler_func=lr_scheduler_func, epoch=epoch)

            # training for one epoch
            total_loss = 0
            total_accuracy = 0
            pbar = tqdm(total=step_train, desc=f'Epoch {epoch + 1}/{self.args.epoch}', postfix=dict, mininterval=0.3)
            for iteration, batch in enumerate(train_dataloader):
                if iteration >= step_train:
                    break
                images, labels = batch
                # update device for data
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                if not self.args.fp16:
                    outputs = self.model_train(images)
                    logits = outputs.logits.to(self.device)
                    # 反向传播
                    loss_value = nn.CrossEntropyLoss()(logits, labels)
                    loss_value.backward()
                    optimizer.step()
                else:
                    # for fp16
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model_train(images)
                        logits = outputs.logits.to(self.device)
                        loss_value = nn.CrossEntropyLoss()(logits, labels)
                    self.scaler.scale(loss_value).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                total_loss += loss_value.item()
                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
                    total_accuracy += accuracy.item()
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'accuracy': total_accuracy / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

            # saving best model_name_or_path per epoch
            self.model.save_pretrained(self.args.model_trained / (get_cur_time() + f"_epoch_{epoch}_loss_{total_loss}"))

            if total_loss < loss_max:
                self.model.save_pretrained(self.args.model_trained / "best")
                # update max loss
                loss_max = total_loss


if __name__ == "__main__":
    # set args
    arguments = Arguments()

    # reference from https://zhuanlan.zhihu.com/p/458809368
    seed_everything(42)

    logger.info("Loading dataset from: " + str(arguments.dataset_path))
    train_dataset = VitDataSet(args=arguments, data_type="train")
    test_dataset = VitDataSet(args=arguments, data_type="test")

    trainer = Trainer(args=arguments,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset)
    trainer.train()
