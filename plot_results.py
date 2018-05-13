import itertools
import json
import math
import os
import psutil
from typing import Dict, List, Tuple
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

N_RUNS = 25
CONFIDENCE = 0.95
Z_STAR = 1.96


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

            shape, loc, scale = stats.lognorm.fit(result)
            pdf = stats.lognorm.pdf(bins, shape, loc, scale)
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


def get_processing_stats_for_run(df: pd.DataFrame, run_idx: int) \
        -> Tuple[float, float]:
    run_df = df.loc[df['run_id'] == run_idx]
    run_df['processing'] = run_df['server_send'] - run_df['server_recv']
    run_df = run_df.loc[run_df['processing'] > 0]

    return run_df['processing'].mean(), run_df['processing'].std()


def get_uplink_stats_for_run(df: pd.DataFrame, run_idx: int) \
        -> Tuple[float, float]:
    run_df = df.loc[df['run_id'] == run_idx]
    run_df['uplink'] = run_df['server_recv'] - run_df['client_send']
    run_df = run_df.loc[run_df['uplink'] > 0]

    return run_df['uplink'].mean(), run_df['uplink'].std()


def get_downlink_stats_for_run(df: pd.DataFrame, run_idx: int) \
        -> Tuple[float, float]:
    run_df = df.loc[df['run_id'] == run_idx]
    run_df['downlink'] = run_df['client_recv'] - run_df['server_send']
    run_df = run_df.loc[run_df['downlink'] > 0]

    return run_df['downlink'].mean(), run_df['downlink'].std()


def plot_avg_times_runsample(experiments: Dict) -> None:
    root_dir = os.getcwd()

    up = []
    down = []
    proc = []
    up_err = []
    down_err = []
    proc_err = []

    for exp_dir in experiments.values():
        # iterate over experiments
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')
        os.chdir(root_dir)

        processing = list(itertools.starmap(
            get_processing_stats_for_run,
            zip(
                itertools.repeat(data),
                range(N_RUNS)
            )
        ))

        uplink = list(itertools.starmap(
            get_uplink_stats_for_run,
            zip(
                itertools.repeat(data),
                range(N_RUNS)
            )
        ))

        downlink = list(itertools.starmap(
            get_downlink_stats_for_run,
            zip(
                itertools.repeat(data),
                range(N_RUNS)
            )
        ))

        proc_avg = mean([x[0] for x in processing])
        up_avg = mean([x[0] for x in uplink])
        down_avg = mean([x[0] for x in downlink])

        proc_std = mean([x[1] for x in processing])
        up_std = mean([x[1] for x in uplink])
        down_std = mean([x[1] for x in downlink])

        # proc_conf = Z_STAR * (proc_std / math.sqrt(N_RUNS))
        # up_conf = Z_STAR * (up_std / math.sqrt(N_RUNS))
        # down_conf = Z_STAR * (down_std / math.sqrt(N_RUNS))
        proc_conf = stats.norm.interval(CONFIDENCE,
                                        loc=proc_avg,
                                        scale=proc_std / math.sqrt(N_RUNS))
        up_conf = stats.norm.interval(CONFIDENCE,
                                      loc=up_avg,
                                      scale=up_std / math.sqrt(N_RUNS))
        down_conf = stats.norm.interval(CONFIDENCE,
                                        loc=down_avg,
                                        scale=down_std / math.sqrt(N_RUNS))

        proc_conf = [abs(x - proc_avg) for x in proc_conf]
        up_conf = [abs(x - up_avg) for x in up_conf]
        down_conf = [abs(x - down_avg) for x in down_conf]

        up.append(up_avg)
        down.append(down_avg)
        proc.append(proc_avg)

        up_err.append(up_conf)
        down_err.append(down_conf)
        proc_err.append(proc_conf)

    # turn error lists from Nx2 to 2XN for plotting
    up_err = list(map(list, zip(*up_err)))
    down_err = list(map(list, zip(*down_err)))
    proc_err = list(map(list, zip(*proc_err)))

    # plot side by side
    bar_width = 0.3
    r1 = np.arange(len(experiments))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig, ax = plt.subplots()
    rect1 = ax.bar(r1, up,
                   label='Avg. uplink time',
                   yerr=up_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       label='95% Confidence Int.'
                   )
                   )
    rect2 = ax.bar(r2, proc,
                   label='Avg. processing time',
                   yerr=proc_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       # label='95% Confidence Int.'
                   )
                   )
    rect3 = ax.bar(r3, down,
                   label='Avg. downlink time',
                   yerr=down_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       # label='95% Confidence Int.'
                   )
                   )

    autolabel(ax, rect1)
    autolabel(ax, rect2)
    autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Add xticks on the middle of the group bars
    plt.title('Time comparison (each run is a sample)')
    plt.xlabel('Number of clients', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(experiments))],
               experiments.keys())

    plt.tight_layout()
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )

    plt.show()


def plot_avg_times_framesample(experiments: Dict) -> None:
    root_dir = os.getcwd()

    up = []
    down = []
    proc = []
    up_err = []
    down_err = []
    proc_err = []

    for exp_dir in experiments.values():
        # iterate over experiments
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')
        os.chdir(root_dir)

        data['processing'] = data['server_send'] - data['server_recv']
        data['uplink'] = data['server_recv'] - data['client_send']
        data['downlink'] = data['client_recv'] - data['server_send']

        proc_df = data.loc[data['processing'] > 0]['processing']
        up_df = data.loc[data['uplink'] > 0]['uplink']
        down_df = data.loc[data['downlink'] > 0]['downlink']

        proc_avg = proc_df.mean()
        up_avg = up_df.mean()
        down_avg = down_df.mean()

        proc_std = proc_df.std()
        up_std = up_df.std()
        down_std = down_df.std()

        shape, loc, scale = stats.lognorm.fit(proc_df)
        proc_conf = stats.lognorm \
            .interval(CONFIDENCE,
                      shape,
                      loc=proc_avg,
                      scale=proc_std / math.sqrt(len(proc_df)))

        shape, loc, scale = stats.lognorm.fit(up_df)
        up_conf = stats.lognorm \
            .interval(CONFIDENCE,
                      shape,
                      loc=up_avg,
                      scale=up_std / math.sqrt(len(up_df)))

        shape, loc, scale = stats.lognorm.fit(down_df)
        down_conf = stats.lognorm \
            .interval(CONFIDENCE,
                      shape,
                      loc=down_avg,
                      scale=down_std / math.sqrt(len(down_df)))

        proc_conf = [abs(x - proc_avg) for x in proc_conf]
        up_conf = [abs(x - up_avg) for x in up_conf]
        down_conf = [abs(x - down_avg) for x in down_conf]

        up.append(up_avg)
        down.append(down_avg)
        proc.append(proc_avg)

        up_err.append(up_conf)
        down_err.append(down_conf)
        proc_err.append(proc_conf)

        # turn error lists from Nx2 to 2XN for plotting
    up_err = list(map(list, zip(*up_err)))
    down_err = list(map(list, zip(*down_err)))
    proc_err = list(map(list, zip(*proc_err)))

    # plot side by side
    bar_width = 0.3
    r1 = np.arange(len(experiments))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    fig, ax = plt.subplots()
    rect1 = ax.bar(r1, up,
                   label='Avg. uplink time',
                   yerr=up_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       label='95% Confidence Int.'
                   )
                   )
    rect2 = ax.bar(r2, proc,
                   label='Avg. processing time',
                   yerr=proc_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       # label='95% Confidence Int.'
                   )
                   )
    rect3 = ax.bar(r3, down,
                   label='Avg. downlink time',
                   yerr=down_err,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(
                       ecolor='gray', lw=1,
                       capsize=2, capthick=1,
                       # label='95% Confidence Int.'
                   )
                   )

    autolabel(ax, rect1)
    autolabel(ax, rect2)
    autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Add xticks on the middle of the group bars
    plt.title('Time comparison (each frame is a sample)')
    plt.xlabel('Number of clients', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(experiments))],
               experiments.keys())

    plt.tight_layout()
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )

    plt.show()


def plot_cpu_loads(experiments: Dict) -> None:
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]
    cpu_loads = [get_avg_cpu_load_for_experiment(x) for x in system_data]

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), cpu_loads, label='Average CPU load')
    autolabel(ax, rect)

    ax.set_ylabel('Load [%]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def plot_ram_usage(experiments: Dict) -> None:
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]
    ram_usage = [get_avg_ram_usage_for_experiment(x) for x in system_data]
    ram_usage = [x / float(1024 * 1024 * 1024) for x in ram_usage]

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), ram_usage, label='Average RAM usage')
    autolabel(ax, rect)

    total_mem = psutil.virtual_memory().total / float(1024 * 1024 * 1024)

    # ax.set_ylim([0, total_mem + 3])
    ax.axhline(y=total_mem, color='red', label='Max. available memory')
    ax.set_ylabel('Usage [GiB]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def get_avg_cpu_load_for_experiment(exp_df: pd.DataFrame) -> float:
    samples = exp_df.loc[exp_df['run_start_cutoff'] < exp_df['timestamp']]
    samples = samples.loc[exp_df['timestamp'] < exp_df['run_end_cutoff']]

    return samples['cpu_load'].mean()


def get_avg_ram_usage_for_experiment(exp_df: pd.DataFrame) -> float:
    samples = exp_df.loc[exp_df['run_start_cutoff'] < exp_df['timestamp']]
    samples = samples.loc[exp_df['timestamp'] < exp_df['run_end_cutoff']]

    total_mem = psutil.virtual_memory().total
    return (total_mem - samples['mem_avail']).mean()


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
        '1 Client'  : '1Client_Benchmark',
        '5 Clients' : '5Clients_Benchmark',
        '10 Clients': '10Clients_Benchmark'
    }

    plot_avg_times_runsample(experiments)
    # plot_avg_times_framesample(experiments)
    plot_cpu_loads(experiments)
    plot_ram_usage(experiments)
    plot_time_dist(experiments)
