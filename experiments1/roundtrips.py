import sys

import numpy as np
import pandas as pd


def parse_rt_stats(file):
    flow_data = pd.read_csv(file)
    flow_data_0 = flow_data[flow_data["trajectory"] == 0]

    total_visits = np.sum(flow_data_0["visits"])

    round_trips = flow_data_0["round_trips"][0]

    return total_visits, round_trips

if __name__ == "__main__":

    file = sys.argv[1]

    total_visits, round_trips = parse_rt_stats(file)

    print(f"total visits: {total_visits}")
    print(f"total round trips: {round_trips}")
    print(f"average round trip time: {total_visits / round_trips}")
