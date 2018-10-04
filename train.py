import argparse
import yaml
import visdom
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from models import combined_net,metircs
from models.loss import *
from models.multibox_loss import MultiBoxLoss

def train(cfg){
    # Setup Augmentation
    data_aug = Compose([RandomRatate(10), RandomHorizontallyFilp()])

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = get_data_path(cfg['data']['dataset'])

    t_loader = data_loader(
        data_path,
        is_transform=True,
        spilt=cfg['data']['train_spilt'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug
    )

    v_loader = data_loader(
        data_path,
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
        for(images,labels) in trainloader:
            scheduler.step(i)
            i+=1
            model.train()
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            # ssd : loc, conf // seg : scores
            loc, conf, scores = model(images)
            seg_loss = loss_seg_fn(input=scores, target=labels)
            ssd_loss = loss_ssd_fn(loc, loc_label, conf, conf_label)
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
                for i_val, (images_val, labels_val)...
}








if __name__ == "__main__":
    paser = argparse.ArugmentParser(description="config")
    parser.add_argument(
            "--config",nargs="?",type=str,default="configs/combined_net.yml",help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    train(cfg)

    
