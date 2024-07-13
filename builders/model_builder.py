#from model.DABNet import DABNet


from model.DECENet import DECENet






from pathlib import Path


def build_model(model_name, n_classes):
    #if model_name == 'DABNet':
    #    return DABNet(classes=n_classes)
    
    if model_name == 'DECENet':
        return DECENet(classes = n_classes)
    
   
