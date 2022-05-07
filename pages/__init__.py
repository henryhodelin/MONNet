from .intro import intro_MONNet
from .structure import structure_MONNet
from .training_details import training_description
from .LAntideriv_descript import linear_antiderivative_op_description
from .LAntideriv_data_gen import linear_antiderivative_training_data_generator
from .LAntideriv_training import linear_antiderivative_training
from .LAntideriv_precision import linear_antiderivative_MONNet_precision_analysis
from .LAntideriv_OPP_data_gen import linear_antiderivative_OPP_training_data_generator

pages = {
    "Introduction": intro_MONNet,
    "MONNet Architecture": structure_MONNet,
    "Training Description": training_description,
    "Linear Antiderivative Operator Description": linear_antiderivative_op_description,
    "Linear Antiderivative Operator Data Training Generation":linear_antiderivative_training_data_generator,
    "Linear Antiderivative MONNet Training ":linear_antiderivative_training,
    "Linear Antiderivative MONNet Precision Analysis" :linear_antiderivative_MONNet_precision_analysis,
    "Testing OOP Antiderivative Operator Data Training Generation": linear_antiderivative_OPP_training_data_generator,
    
}