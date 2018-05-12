import json
import os
from typing import Dict, List
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import lognorm

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


def plot_time_dist(experiments: Dict) -> None:
    root_dir = os.getcwd()
    up_results = []
    down_results = []
    proc_results = []
    for exp_name, exp_dir in experiments.items():
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')
        processing = []
        uplink = []
        downlink = []
        for row in data.itertuples():
            proc = row.server_send - row.server_recv
            up = row.server_recv - row.client_send
            down = row.client_recv - row.server_send
            # rtt = row.client_recv - row.client_send
            if proc > 0:
                processing.append(proc)

            if up > 0:
                uplink.append(up)

            if down > 0:
                downlink.append(down)

        os.chdir('..')
        proc_results.append(processing)
        up_results.append(uplink)
        down_results.append(downlink)

    with plt.style.context('ggplot'):

        # processing times

        fig, ax = plt.subplots()
        abs_max = max([max(x) for x in proc_results])
        abs_min = min([min(x) for x in proc_results])
        if abs_min == 0:
            abs_min = 1

        bins = np.logspace(np.log10(abs_min), np.log10(abs_max), 30)

        for i, result in enumerate(proc_results):
            ax.hist(result, bins,
                       label=list(experiments.keys())[i],
                       # norm_hist=True
                       alpha=0.5,
                       density=True)

            shape, loc, scale = lognorm.fit(result)
            pdf = lognorm.pdf(bins, shape, loc, scale)
            ax.plot(bins, pdf,
                       label=list(experiments.keys())[i] + ' PDF')

        plt.title('Processing times')
        ax.set_xscale("log")
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Density')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        # uplink times
        fig, ax = plt.subplots()
        abs_max = max([max(x) for x in up_results])
        abs_min = min([min(x) for x in up_results])
        if abs_min == 0:
            abs_min = 1

        bins = np.logspace(np.log10(abs_min), np.log10(abs_max), 30)
        # bins = np.linspace(abs_min, abs_max, 30)

        for i, result in enumerate(up_results):
            ax.hist(result, bins,
                       label=list(experiments.keys())[i],
                       # norm_hist=True
                       alpha=0.5,
                       density=True)

            # shape, loc, scale = lognorm.fit(result)
            # pdf = lognorm.pdf(bins, shape, loc, scale)
            # ax[1].plot(bins, pdf,
            #            label=list(experiments.keys())[i] + ' PDF')

        plt.title('Uplink times')
        ax.set_xscale("log")
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Density')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()

        # downlink times
        fig, ax = plt.subplots()
        abs_max = max([max(x) for x in down_results])
        abs_min = min([min(x) for x in down_results])
        if abs_min == 0:
            abs_min = 1

        bins = np.logspace(np.log10(abs_min), np.log10(abs_max), 30)
        # bins = np.linspace(abs_min, abs_max, 30)

        for i, result in enumerate(down_results):
            ax.hist(result, bins,
                       label=list(experiments.keys())[i],
                       # norm_hist=True
                       alpha=0.5,
                       density=True)

            # shape, loc, scale = lognorm.fit(result)
            # pdf = lognorm.pdf(bins, shape, loc, scale)
            # ax[1].plot(bins, pdf,
            #            label=list(experiments.keys())[i] + ' PDF')

        plt.title('Downlink times')
        ax.set_xscale("log")
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Density')

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
    plot_time_dist(experiments)
