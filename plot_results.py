import json
import os
from typing import Dict, List
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

N_RUNS = 5


def autolabel(ax: plt.Axes, rects: List[plt.Rectangle], center=False) -> None:
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2.0

        if center:
            y_pos = 0.5 * height + rect.get_y() if height >= 5.0 else \
                rect.get_y()
        else:
            y_pos = 1.01 * height + rect.get_y()
        ax.text(x_pos, y_pos,
                '{:02.2f}'.format(height),
                ha='center', va='bottom', weight='bold')


def plot_proctime_dist(experiments: Dict) -> None:
    root_dir = os.getcwd()
    data = []
    for exp_name, exp_dir in experiments.items():
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')
        processing = []
        for row in data.itertuples():
            proc = row.server_send - row.server_recv
            # rtt = row.client_recv - row.client_send
            if proc > 0:
                processing.append(proc)
        os.chdir('..')
        data.append(processing)

    fig, ax = plt.subplots()
    plt.style.use('seaborn-deep')

    bins = np.linspace(0, 1000, num=50)
    ax.hist(data, bins, label=experiments.keys())
    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def plot_avg_times(experiments: Dict) -> None:
    uplink_avg = []
    downlink_avg = []
    processing_avg = []

    up_error = []
    down_error = []
    proc_error = []

    root_dir = os.getcwd()
    for exp_dir in experiments.values():
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')

        uplink = []
        downlink = []
        processing = []
        for row in data.itertuples():
            up = row.server_recv - row.client_send
            proc = row.server_send - row.server_recv
            down = row.client_recv - row.server_send
            # rtt = row.client_recv - row.client_send

            if up > 0:
                uplink.append(up)
            if down > 0:
                downlink.append(down)
            if proc > 0:
                processing.append(proc)
        os.chdir('..')

        uplink_avg.append(mean(uplink))
        downlink_avg.append(mean(downlink))
        processing_avg.append(mean(processing))

        up_error.append(stdev(uplink))
        down_error.append(stdev(downlink))
        proc_error.append(stdev(processing))

    bar_width = 0.3
    r1 = np.arange(len(experiments))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig, ax = plt.subplots()
    rect1 = ax.bar(r1, uplink_avg,
                   label='Avg. uplink time',
                   yerr=up_error,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
                   )
    rect2 = ax.bar(r2, processing_avg,
                   label='Avg. processing time',
                   yerr=proc_error,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
                   )
    rect3 = ax.bar(r3, downlink_avg,
                   label='Avg. downlink time',
                   yerr=down_error,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1)
                   )

    autolabel(ax, rect1)
    autolabel(ax, rect2)
    autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Add xticks on the middle of the group bars
    plt.xlabel('Number of clients', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(experiments))],
               experiments.keys())

    plt.tight_layout()
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )

    plt.show()


def plot_cpu_loads(cpu_loads: List[float],
                   experiment_names: List[str]) -> None:
    assert len(cpu_loads) == len(experiment_names)

    fig, ax = plt.subplots()
    rect = ax.bar(experiment_names, cpu_loads, label='Average CPU load')
    autolabel(ax, rect)

    ax.set_ylabel('Load [%]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def get_avg_cpu_load_for_experiment(experiment_df: pd.DataFrame) -> float:
    count = 0
    total_sum = 0

    for _, row in experiment_df.iterrows():
        if row['run_start_cutoff'] < row['timestamp'] < row['run_end_cutoff']:
            count += 1
            total_sum += row['cpu_load']

    return total_sum / (1.0 * count)


def load_data_for_experiment(experiment_id) -> Dict:
    os.chdir(experiment_id)
    with open('total_stats.json', 'r') as f:
        os.chdir('..')
        return json.load(f)


def load_system_data_for_experiment(experiment_id) -> pd.DataFrame:
    os.chdir(experiment_id)
    df = pd.read_csv('total_system_stats.csv')
    os.chdir('..')
    return df


if __name__ == '__main__':
    experiments = {
        '1 Client'  : '1Client_IdealBenchmark',
        '5 Clients' : '5Clients_IdealBenchmark',
        '10 Clients': '10Clients_IdealBenchmark'
    }

    exp_data = [load_data_for_experiment(x) for x in experiments.values()]
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]

    cpu_loads = [get_avg_cpu_load_for_experiment(x) for x in system_data]

    plot_avg_times(experiments)
    plot_cpu_loads(cpu_loads, list(experiments.keys()))
    plot_proctime_dist(experiments)
