import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.getcwd())
from utility.learning_utils import *
from data_loader.loader import create_tasks, get_dataset_name, get_current_date
from batch_cil_experiment.cil import perform_ssl_cil_tasks, get_models
import argparse
import torch
from byol_pytorch import BYOL
from models.AlexNet_cl import AlexNet_encoder
from models.multi_head_classifier import Multi_head
import os, wandb

NUMPY_SEED = 24
PYTORCH_SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cil tasks')
    parser.add_argument('--dataset_name', type=str, help='the dataset name')
    parser.add_argument('--cil_step', type=str, help='number of classes for the each incremental learning')
    parser.add_argument('--cil_start', type=str, help='the list classes for learning in the first task')
    parser.add_argument('--lr', type=str, default='0.0001', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--epochs', type=str, default='20', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--report_step', type=str, default='100', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--reset_model', type=str, default='True', help='reset model at each new task')
    parser.add_argument('--ssl', nargs="+", default=[], help='select with ssl to train as auxilary task, option are byol, rotnet and ae')
    parser.add_argument('--ratio', type=str, default=1.0, help='ratio on total training time where supervised loss vs self-supervised in present')
    parser.add_argument('--batch_size', type=str, default=16, help='ratio on total training time where supervised loss vs self-supervised in present')
    args = parser.parse_args()

    args.reset_model = False if args.reset_model == 'False' else True

    dataset_name = get_dataset_name(args.dataset_name)
    cil_start = list(range(int(args.cil_start)))
    cil_step = int(args.cil_step)
    lr = float(args.lr)
    epochs = int(args.epochs)
    report_step = int(args.report_step)
    ssl_methods = set(args.ssl)
    ratio = float(args.ratio)
    batch_size = int(args.batch_size)

    torch.manual_seed(PYTORCH_SEED)
    np.random.seed(NUMPY_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{Color.GREEN.value}Device:{device} {Color.END.value}')
    d = get_current_date()


    ssl_str = ""
    for ssl in args.ssl:
        ssl_str += " " + str(ssl)
    name_run = args.dataset_name + " - " + ssl_str + " - epoch:" +  args.epochs + " - ratio:" +  args.ratio

    #wandb init
    os.environ["WANDB_MODE"] = "dryrun"
    run = wandb.init(
        project="ssl-cl",
        entity="ssl-cl",
        dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
        name = name_run
    )
    wandb.run.summary["tensorflow_seed"] = PYTORCH_SEED
    wandb.run.summary["numpy_seed"] = NUMPY_SEED
    wandb.config.update(args)

    #define the base encoder, the common model across supervised and self-supervised
    encoder = AlexNet_encoder()
    encoder.to(device)

    #define the multihead classifier, that has one output layer per task id
    multi_head = Multi_head(cil_step)
    multi_head.to(device)


    # add list of ssl models to pass for training
    ssl_models = []

    for method in ssl_methods:

        if 'byol' in ssl_methods:
            byol_model = BYOL(
                encoder,
                image_size=32,
                hidden_layer = -1
            )
            ssl_models.append(byol_model)

        if 'xxx' in ssl_methods:  # add extra ssl model here
            pass
        if 'yyy' in ssl_methods:  # add extra ssl model here
            pass

        else :
            ssl_models.append(None)

    ssl_dict = dict(zip(ssl_methods,ssl_models))


    # tasks = [[0, 1], [2], [3]]
    # epochs = 2, lr = 0.0001, report_step = 20
    tasks = create_tasks(dataset_name=dataset_name, cil_step=cil_step, cil_start=cil_start)
    perform_ssl_cil_tasks(tasks, (encoder, multi_head), dataset_name=dataset_name, epochs=epochs, lr=lr,
                      report_step=report_step, ssl_dict = ssl_dict, ratio = ratio, batch_size = batch_size)
    print("Done!")

####### for ssl
# --dataset_name cifar10 --cil_step 2 --cil_start 2 --epochs 50 --ssl byol --report_step 100
