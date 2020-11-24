import pandas as pd
from collections import deque
import csv
import time
import torch
from tqdm import tqdm


def abides_to_tensor(log_file = 'ORDERBOOK_ABM_FULL', output_file='orderbook.csv'):
    # Read log file
    df = pd.read_pickle(log_file)

    order_book_history = []
    for timestamp, row in tqdm(df.iterrows(), total=len(df)):
        # Early stop
        if len(order_book_history) >= 360000:
            break
        # Filter out empty entries
        row = row[row != 0]
        buys = row[row < 0]
        sells = row[row > 0]
        # Check for partial order book and skip
        if len(buys) < 10 or len(sells) < 10:
            continue
        else:
            order_book = []

            timestamp = time.mktime(timestamp.timetuple())
            # Populate order book
            order_book.append(timestamp)
            for sell, buy in zip(sells[:10].items(), buys[:-11:-1].abs().items()):
                order_book.extend(sell)
                order_book.extend(buy)
            order_book_history.append(order_book)
    t = torch.tensor(order_book_history, dtype=torch.float64)
    # t[:, 1::2] *= 1e-4  # change prices to dollar ammounts
    torch.save(t, 'data/F/long_data/00000000')


abides_to_tensor('../abides/log/rmsc03_two_hour/ORDERBOOK_F_FREQ_10L.bz2')