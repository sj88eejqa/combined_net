import argparse
import yaml
import visdom
import torch
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
import torchvision.transforms as transforms
from models import combined_net,metircs
from models.loss import *
from models.multibox_loss import MultiBoxLoss

def train(cfg){
    # Setup Augmentation
    data_aug = Compose([RandomRatate(10), RandomHorizontallyFilp()])

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = get_data_path(cfg['data']['dataset'])

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    t_loader = data_loader(
        root=data_path,
        detect_list="ImageSets/Detection",
        is_train=True,
        trainform=transform,
        is_transform=True,
        spilt=cfg['data']['train_spilt'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug
    )

    v_loader = data_loader(
        root=data_path,
        detect_list="ImageSets/Detection",
        is_train=False,
        tranform=transform,
        is_transform=True,
        spilt=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=cfg['training']['batch_size'],num_workers=2,shuffle=True
    )

    valloader = data.DataLoader(v_loader, batch_size=cfg['training']['batch_size'],num_workers=2)

    # Setup Metrics
    running_metrics_val = metrics.runningScore(n_classes)

    # Setup visdom for visualization, default is false
    if cfg['training']['visdom']:
        vis = visdom.Visdom()

        loss_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(
                xlabel="minibatches",
                ylabel="Loss",
                title="Training Loss",
                legend=["Loss"],
            ),
        )
    
    #Setup model
    model = combined_net(backbone="resnext50",image_shape=(256,256),n_class=22).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg['training']['lr_rate'],
                            momentum=cfg['training']['momentum'],
                            weight_decay=cfg['training']['weight_decay']
    )

    loss_seg_fn = cross_entropy2d #multi_scale_cross_entropy2d
    loss_ssd_fn = MultiBoxlLoss()
    
    start_iter=0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            print(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_iter = checkpoint["epoch"]
            print(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            # if cfg['model']['arch']=="pspnet":
            #     caffemodel_dir_path = "/home/timlu/Documents/research/PSPNet/PSPNet/evaluation/model"
            #     model.load_pretrained_model(
            #         model_path=os.path.join(caffemodel_dir_path,"pspnet101_VOC2012.caffemodel")
            #     )
            print("No checkpoint found at '{}'".format(cfg['training']['resume']))
            print("No pretrained model now.")

    best_iou = -100.0
    i = start_iter

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda i: (1-i/cfg['training']['train_iters'])**cfg['training']['power']])

    while i <= cfg['training']['train_iters']:
        for(images,seg_labels,loc_labels,conf_labels) in trainloader:
            scheduler.step(i)
            i+=1
            model.train()
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            # ssd : loc, conf // seg : scores
            loc, conf, scores = model(images)
            seg_loss = loss_seg_fn(input=scores, target=seg_labels)
            ssd_loss = loss_ssd_fn(loc, loc_labels, conf, conf_labels)
            loss = seg_loss+ssd_loss
            loss.backward()
            optimizer.step()
    
            if cfg['training']['visdom']:
                vis.line(
                    X=torch.ones((1,1)).cpu()*i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    updata="append",
                )
            if (i+1)%cfg['training']['print_interval']==0:
                print(
                       "Iter [%d/%d] Loss: %.4f LR: %.8f" % (i+1, cfg['training']['train_iters'], loss.item(), scheduler.get_lr()[-1])
                )
            if (i+1)%cfg['training']['val_interval']==0:
                model.eval()
                ssd_loss = 0
                for i_val, (images_val, seg_val, loc_val, conf_val) in tqdm(enumerate(valloader)):
                    images_val = images_val.cuda()
                    seg_val = seg_val.cuda()
                    loc_val = loc_val.cuda()
                    conf_val = conf_val.cuda()

                    loc, conf, scores = model(images_val)
                    # segmentation    
                    pred = scores.data.max(1)[1].cpu().numpy()
                    gt = seg_val.data.cpu().numpy()
                    running_metrics_val.update(gt, pred)
                    # ssd-detect
                    loss = loss_ssd_fn(loc, loc_val, conf,conf_val)
                    ssd_loss += loss.data[0]
                    print('%.3f %.3f' % (loss.data[0],test_loss/(i_val+1)))

                score, class_iou = running_metrics_val.get_scores()
                for k,v in score.items():
                    print(k,v)
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    torch.save(state, "{}_{}_best_model.pkl".format(cfg['model']['arch'],
                                                                    cfg['data']['dataset']))


if __name__ == "__main__":
    paser = argparse.ArugmentParser(description="config")
    parser.add_argument(
            "--config",nargs="?",type=str,default="configs/combined_net.yml",help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    train(cfg)

    
