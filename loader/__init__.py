import loader.pascal_voc_loader as vocloader
import loader.ade20k_loader as ade20kloader
from loader.datagen import ListDataset
from torch.utils import data

def get_loader(name):
    class seg_detect_loader(data.Dataset):
        def __init__(self,
            root,
            detect_list,
            is_train=False,
            transform,
            split='train_aug',
            is_transform=False,
            img_size=512,
            augmentation=None,
            img_norm=True,
        ):
            self.seg_loader = {
                "pascal":vocloader.pascalVOCLoader,
                "ade20k":ade20kloader.ADE20KLoader
            }[name](root, is_transform, split, img_size, augmentation,img_norm)
            detect_list = pjoin(root,detect_list,("voc12_train.txt" if is_train else "voc12_test.txt"))
            self.detect_loader = ListDataset(root,detect_list,is_train,transform)

        def __len__(self):
            assert len(self.seg_loder)==len(self.detect_loader), "two loader's len must be same"
            return len(self.seg_loder)

        def __getitem__(self,idx):
            img, seg_label = self.seg_loader[idx]
            _, loc_label, conf_label = self.detect_loader[idx]
            return img, seg_label, loc_label, conf_label

    return seg_detect_loader

def get_data_path(name, config_file="config.josn"):
    data = json.load(open(config_file))
    return data[name]["data_path"]
