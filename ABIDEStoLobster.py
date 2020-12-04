import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
import argparse
from glob import glob
import os


def abides_to_tensor(in_dir='../abides/log/rmsc03_two_hour/20191001', out_dir='data/F/long_data', idx=0):
    # Read log file
    log_file = glob(os.path.join(in_dir, 'ORDERBOOK_*.bz2'))[0]
    df = pd.read_pickle(log_file)
    # df = df.head()
    progress = tqdm(total=len(df))
    # Convert to numpy for better performance
    in_array = df.to_numpy()
    header = df.columns.to_numpy()
    timestamp = df.index.to_numpy()
    # Initialize result array
    out_array = np.ndarray((360000, 41))
    # Fill result array. Assuming input has enough data
    cur = 0
    for i, row in enumerate(in_array):
        # Early stop
        if i > 360000:
            break
        # Find buy orders
        index = row != 0
        row = row[index]
        buys_index = row < 0
        buys = row[buys_index]
        # Skip if there isn't enough orders
        if len(buys) < 10 or len(row) - len(buys) < 10:
            continue
        # Find sell orders
        sells_index = row > 0
        sells = row[sells_index]
        # Get prices
        buy_prices, sell_prices = header[index][buys_index], header[index][sells_index]
        # Re-order and combine
        orders = np.array(list(zip(buy_prices[:-11:-1], abs(buys[:-11:-1]), sell_prices[:10], sells[:10]))).ravel()
        # Fill result array
        out_array[cur][0] = timestamp[i]
        out_array[cur][1:] = orders

        cur += 1
        progress.update()

    t = torch.tensor(out_array).float()
    t[:, 1::2] *= 1e-4  # change prices to dollar ammounts
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(t, f'{out_dir}/{idx:08}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Abides logs into tensors.')

    parser.add_argument('-l',
                        '--log',
                        required=True,
                        help='Path to the directory with sub-directory containing logs for each day')
    args = parser.parse_args()

    paths = glob(os.path.join(args.log, '*'))
    for i, path in enumerate(paths):
        abides_to_tensor(path, idx=i)
