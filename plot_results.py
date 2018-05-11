import json
import os
from collections import namedtuple
from typing import Dict, List

import psutil
import matplotlib.pyplot as plt
import pandas as pd

N_RUNS = 5

AvgTimes = namedtuple('AvgTimes', ['uplink', 'downlink', 'processing'])


def autolabel(ax: plt.Axes, rects: List[plt.Rectangle]) -> None:
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2.0
        y_pos = 0.5 * height + rect.get_y() if height >= 5.0 else \
            rect.get_y()
        ax.text(x_pos, y_pos,
                '{:02.2f}'.format(height),
                ha='center', va='bottom', weight='bold')


def get_avg_times(data: Dict) -> AvgTimes:
    total_up = 0
    total_down = 0
    total_proc = 0
    count_up = 0
    count_down = 0
    count_proc = 0
    for run, r_data in data.items():
        for client, data in r_data.items():
            total_up += (data['run_results']['avg_up'] *
                         data['run_results']['count_up'])
            count_up += data['run_results']['count_up']

            total_down += (data['run_results']['avg_down'] *
                           data['run_results']['count_down'])
            count_down += data['run_results']['count_down']

            total_proc += (data['run_results']['avg_proc'] *
                           data['run_results']['count_proc'])
            count_proc += data['run_results']['count_proc']

    avg_up = total_up / count_up
    avg_down = total_down / count_down
    avg_proc = total_proc / count_proc

    return AvgTimes(avg_up, avg_down, avg_proc)


def plot_avg_times(avg_times: List[AvgTimes],
                   experiment_names: List[str]) -> None:
    assert len(experiment_names) == len(avg_times)

    uplink = [t.uplink for t in avg_times]
    downlink = [t.downlink for t in avg_times]
    proc = [t.processing for t in avg_times]

    fig, ax = plt.subplots()
    rect1 = ax.bar(experiment_names, uplink, label='Avg. uplink time')
    rect2 = ax.bar(experiment_names, proc,
                   bottom=uplink, label='Avg. processing time')
    rect3 = ax.bar(experiment_names, downlink,
                   bottom=[x + y for x, y in zip(uplink, proc)],
                   label='Avg. downlink time')

    autolabel(ax, rect1)
    autolabel(ax, rect2)
    autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )

    plt.show()


def plot_cpu_load(experiment_df: List[pd.DataFrame]) -> None:
    pass



def load_data_for_experiment(experiment_id) -> Dict:
    os.chdir(experiment_id)
    with open('total_stats.json', 'r') as f:
        os.chdir('..')
        return json.load(f)


if __name__ == '__main__':
    experiments = {
        '1 Client': '1Client_IdealBenchmark',
        '5 Clients': '5Clients_IdealBenchmark',
        '10 Clients': '10Clients_IdealBenchmark'
    }

    exp_data = [load_data_for_experiment(x) for x in experiments.values()]
    avg_times = [get_avg_times(data) for data in exp_data]
    plot_avg_times(avg_times, list(experiments.keys()))
