# import os
# import sys

# if os.path.abspath('..') not in sys.path:
#     sys.path.append(os.path.abspath('..'))

# from graphslim.config import cli
# from graphslim.dataset import *
# from graphslim.evaluation import Evaluator

# if __name__ == '__main__':
#     args = cli(standalone_mode=False)
#     data = get_dataset(args.dataset, args)
#     if args.eval_whole:
#         evaluator = Evaluator(args)
#         evaluator.MIA_evaluate(data, model_type=args.eval_model, reduced=False)
#     else:
#         if args.attack is not None:
#             data = attack(data, args)
#             args.save_path = f'checkpoints'
#         evaluator = Evaluator(args)
#         evaluator.evaluate(data, model_type=args.eval_model)


import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import cli
from graphslim.dataset import *
from graphslim.evaluation import Evaluator, PropertyEvaluator

if __name__ == '__main__':
    args = cli(standalone_mode=False)
    data = get_dataset(args.dataset, args)
    if args.eval_whole:
        # evaluator = PropertyEvaluator(args)
        evaluator = Evaluator(args)
        evaluator.evaluate(data, reduced=False, model_type='GCN')
   
    else:
        if args.attack is not None:
            data = attack(data, args)
            args.save_path = f'checkpoints'
        # evaluator = PropertyEvaluator(data, args, reduced=True)
        #evaluator = PropertyEvaluator(args)
        evaluator = Evaluator(args)
        evaluator.evaluate(data, reduced=True, model_type='GCN')

