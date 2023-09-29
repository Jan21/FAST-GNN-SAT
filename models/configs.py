from .neurosat_architecture import NeuroSAT

models_with_args = {

    'NeuroSAT': {
            'model_class': NeuroSAT,
            'model_args':{
            'd': 16,
            'final_reducer':'mean',
            }
    }
                       
    }
