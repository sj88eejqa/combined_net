import loader.pascal_voc_loader as vocloader
import loader.ade20k_loader as ade20kloader

def get_loader(name):

    return {
        "pascal":vocloader.pascalVOCLoader,
        "ade20k":ade20kloader.ADE20KLoader
    }[name]

def get_data_path(name, config_file="config.josn"):
    data = json.load(open(config_file))
    return data[name]["data_path"]