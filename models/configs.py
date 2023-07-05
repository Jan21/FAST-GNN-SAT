from .neurosat_architecture import NeuroSAT

models_with_args = {

    'NeuroSAT': {
            'model_class': NeuroSAT,
            'model_args':{
            'd': 16,
            'n_msg_layers':3,
            'n_vote_layers':3,
            'n_rounds':10,
            'mlp_transfer_fn':'relu',
            'final_reducer':'mean',
            'lstm':'standard',
            }
    }
                       
    }
