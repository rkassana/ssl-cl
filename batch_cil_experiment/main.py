import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.getcwd())
from utility.learning_utils import *
from data_loader.loader import create_tasks, get_dataset_name, get_current_date, DataSetName
from batch_cil_experiment.cil import perform_cil_tasks, get_models
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cil tasks')
    parser.add_argument('--dataset_name', type=str, help='the dataset name', default='cifar10')
    parser.add_argument('--cil_step', type=str, help='number of classes for the each incremental learning', default='2')
    parser.add_argument('--cil_start', type=str, help='the list classes for learning in the first task', default='2')
    parser.add_argument('--lr', type=str, default='0.0001', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--epochs', type=str, default='20', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--report_step', type=str, default='100', required=False, help='number of classes for the each incremental learning')
    parser.add_argument('--reset_model', type=str, default='True', help='reset model at each new task')
    parser.add_argument('--approach', default='dll')
    args = parser.parse_args()

    args.reset_model = False if args.reset_model == 'False' else True

    dataset_name = DataSetName.CIFAR10
    cil_start = list(range(int(args.cil_start)))
    cil_step = int(args.cil_step)
    lr = float(args.lr)
    epochs = int(args.epochs)
    report_step = int(args.report_step)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{Color.GREEN.value}Device:{device} {Color.END.value}')
    d = get_current_date()
    model = get_models(dataset_name=dataset_name)

    # tasks = [[0, 1], [2], [3]]
    # epochs = 2, lr = 0.0001, report_step = 20
    tasks = create_tasks(dataset_name=dataset_name, cil_step=cil_step, cil_start=cil_start)
    perform_cil_tasks(tasks, model, dataset_name=dataset_name, epochs=epochs, lr=lr,
                      report_step=report_step, approach=args.approach)
    print("Done!")

####### for dll
# --dataset_name cifar10 --cil_step 2 --cil_start 2 --epochs 50 --approach dll --report_step 100
