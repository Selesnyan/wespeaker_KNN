from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb-path')
    args = parser.parse_args()

    ea = event_accumulator.EventAccumulator(args.tb_path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    min_val_ind = np.argmin(list(map(lambda x: x.value, ea.Scalars('der/val_ep'))))
    min_eval_der = ea.Scalars('der/eval_ep')[min_val_ind].value

    print('_'.join(args.tb_path.split('/')), min_eval_der)


if __name__ == '__main__':
    main()

