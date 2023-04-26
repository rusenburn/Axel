import torch as T

def seconds_to_HMS(seconds:float):
    hours , remainder= divmod(seconds,3600)
    minutes ,seconds =  divmod(remainder,60)
    return hours,minutes,seconds

def calculate_explained_variance(prediction:T.Tensor,target:T.Tensor)->float:
        assert prediction.shape == target.shape
        prediction , target = prediction.flatten(),target.flatten()
        target_var = target.var()+1e-8
        unexplained_var_ratio = (target-prediction).var()/target_var
        explained_var_ratio = 1 - unexplained_var_ratio
        if isinstance(explained_var_ratio,T.Tensor):
            return explained_var_ratio.cpu().item()
        else:
            return explained_var_ratio