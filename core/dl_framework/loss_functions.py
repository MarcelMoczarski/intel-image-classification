import torch.nn.functional as F

def cross_entropy(x, y):
    return F.cross_entropy(x, y)
    
def l2_regularization(loss, model, l2_lambda):
    l2_norm = sum(params.pow(2.0).sum() for params in model.parameters())
    loss = loss + l2_lambda * l2_norm
    return loss

def get_regularization(config_file):
    regularization = None
    params = None
    regularization_name = [r for r in config_file.keys() if "regularization" in r]
    if regularization_name:
        regularization_func_name = regularization_name[0].split("_", 1)[1]
        regularization = globals()[regularization_func_name]
        params = config_file[regularization_name[0]]
        if type(params) is not list:
            params = [params]
    return regularization, params

class loss_function():
    def __init__(self, config_file, model):
        self.loss_function = globals()[config_file["g_loss_func"]]
        self.regularization, self.regularization_params = get_regularization(config_file)

        self.model = model

    def calc(self, out, yb):
        loss = self.loss_function(out, yb)
        if self.regularization:
            loss = self.regularization(loss, self.model, *self.regularization_params)
        return loss