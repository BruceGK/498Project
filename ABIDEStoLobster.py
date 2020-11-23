import pandas as pd
from collections import deque
import csv
import time
import torch

def convertABIDEStoLobster(log_file = 'ORDERBOOK_ABM_FULL', output_file='orderbook.csv'):
    df = pd.read_pickle(log_file)
    best_sells = deque()
    best_buys = deque()
    order_book_history = []
    for index, row in df.iterrows():
        for j in df.columns:
            if row[j] > 0:
                best_sells.append((j, abs(row[j])))
            elif row[j] < 0:
                best_buys.append((j, abs(row[j])))
        if len(best_sells) >= 10 and len(best_buys) >= 10:
            timestamp = time.mktime(index.timetuple())
            order_book_history.append((timestamp, sorted(best_sells, reverse=False)[:10], sorted(best_buys, reverse=True)[:10]))
        best_sells = deque()
        best_buys = deque()
    
    with open(output_file, 'w', newline='') as csvfile:
        OBwriter = csv.writer(csvfile, delimiter=',')
        for i in range(len(order_book_history)):
            row = []
            sells = order_book_history[i][0]
            buys = order_book_history[i][1]
            row.append(order_book_history[i][0])
            for k in range(10):
                row.append(sells[k][0])
                row.append(sells[k][1])
                row.append(buys[k][0])
                row.append(buys[k][1])
            OBwriter.writerow(row)
    OB = pd.read_csv(output_file, header = None)
    return torch.tensor(OB.values)

def abides_to_tensor(log_file = 'ORDERBOOK_ABM_FULL', output_file='orderbook.csv'):
    # Read log file
    df = pd.read_pickle(log_file)

    order_book_history = []
    for timestamp, row in df.iterrows():
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
            # t = torch.tensor(order_book)

abides_to_tensor('ORDERBOOK_ABM_FULL.bz2')