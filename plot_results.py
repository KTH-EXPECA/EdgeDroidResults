import itertools
import json
import math
import os
from statistics import mean
from typing import Dict, List, Tuple, NamedTuple
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy import stats

N_RUNS = 25
CONFIDENCE = 0.95
Z_STAR = 1.96
SAMPLE_FACTOR = 10

Stats = NamedTuple('Stats', [('mean', float),
                             ('std', float),
                             ('conf_lower', float),
                             ('conf_upper', float)])

ExperimentTimes = NamedTuple('ExperimentTimes',
                             [('processing', Stats),
                              ('uplink', Stats),
                              ('downlink', Stats)])


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


def plot_avg_times_frames(experiments: Dict, feedback: bool = False) -> None:
    root_dir = os.getcwd()

    stats = []

    for exp_dir in experiments.values():
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv', index_col=0)
        os.chdir(root_dir)

        stats.append(frames_stats(data, feedback=feedback))

    processing_means = [s.processing.mean for s in stats]
    processing_errors = [[s.processing.mean - s.processing.conf_lower
                          for s in stats],
                         [s.processing.conf_upper -
                          s.processing.mean
                          for s in stats]]

    uplink_means = [s.uplink.mean for s in stats]
    uplink_errors = [[s.uplink.mean - s.uplink.conf_lower
                      for s in stats],
                     [s.uplink.conf_upper -
                      s.uplink.mean
                      for s in stats]]

    downlink_means = [s.downlink.mean for s in stats]
    downlink_errors = [[s.downlink.mean - s.downlink.conf_lower
                        for s in stats],
                       [s.downlink.conf_upper -
                        s.downlink.mean
                        for s in stats]]

    bar_width = 0.3
    r1 = np.arange(len(experiments))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    errorbar_opts = dict(
        ecolor='darkorange',
        lw=2, alpha=1.0,
        capsize=0, capthick=1
    )

    fig, ax = plt.subplots()
    rect1 = ax.bar(r1, uplink_means,
                   label='Avg. uplink time',
                   yerr=uplink_errors,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=dict(errorbar_opts, label='95% Confidence Int.')
                   )
    rect2 = ax.bar(r2, processing_means,
                   label='Avg. processing time',
                   yerr=processing_errors,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=errorbar_opts
                   )
    rect3 = ax.bar(r3, downlink_means,
                   label='Avg. downlink time',
                   yerr=downlink_errors,
                   width=bar_width,
                   edgecolor='white',
                   error_kw=errorbar_opts
                   )

    autolabel(ax, rect1)
    autolabel(ax, rect2)
    autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Add xticks on the middle of the group bars
    if feedback:
        plt.title('Time statistics for frames w/ feedback')
    else:
        plt.title('Time statistics for frames w/o feedback')
    plt.xlabel('Number of clients', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(experiments))],
               experiments.keys())

    plt.tight_layout()
    plt.show()


def frames_stats(data: pd.DataFrame, feedback: bool = False) -> ExperimentTimes:
    if feedback:
        frame_data = data.loc[data['feedback']]
    else:
        # filter only frames without feedback
        frame_data = data.loc[~data['feedback']]

    frame_data['processing'] = \
        frame_data['server_send'] - frame_data['server_recv']
    frame_data['uplink'] = \
        frame_data['server_recv'] - frame_data['client_send']
    frame_data['downlink'] = \
        frame_data['client_recv'] - frame_data['server_send']

    # only count frames with positive values (time can't be negative)
    frame_data = frame_data.loc[frame_data['processing'] > 0]
    frame_data = frame_data.loc[frame_data['uplink'] > 0]
    frame_data = frame_data.loc[frame_data['downlink'] > 0]

    # if not feedback:
    #     # finally, only consider every 3rd frame for non-feedback frames
    #     frame_data = frame_data.iloc[::3, :]

    if feedback:
        samples = [frame_data.loc[frame_data['run_id'] == run_id].sample()
                   for run_id in range(N_RUNS)]
    else:
        # find number of clients
        n_clients = frame_data['client_id'].max() + 1
        # take SAMPLE_FACTOR samples per client per run
        samples = []
        for run_id in range(N_RUNS):
            run_data = frame_data.loc[frame_data['run_id'] == run_id]
            for client_id in range(n_clients):
                client_data = run_data.loc[run_data['client_id'] == client_id]
                samples.append(client_data.sample(n=SAMPLE_FACTOR))

    samples = pd.concat(samples)

    # stats for processing times:
    proc_mean = samples['processing'].mean()
    proc_std = samples['processing'].std()
    proc_conf = stats.norm.interval(CONFIDENCE,
                                    loc=proc_mean,
                                    scale=proc_std / math.sqrt(N_RUNS))
    proc_stats = Stats(proc_mean, proc_std, *proc_conf)

    # stats for uplink times:
    up_mean = samples['uplink'].mean()
    up_std = samples['uplink'].std()
    up_conf = stats.norm.interval(CONFIDENCE,
                                  loc=up_mean,
                                  scale=up_std / math.sqrt(N_RUNS))
    up_stats = Stats(up_mean, up_std, *up_conf)

    # stats for downlink times:
    down_mean = samples['downlink'].mean()
    down_std = samples['downlink'].std()
    down_conf = stats.norm.interval(CONFIDENCE,
                                    loc=down_mean,
                                    scale=down_std / math.sqrt(N_RUNS))
    down_stats = Stats(down_mean, down_std, *down_conf)

    return ExperimentTimes(proc_stats, up_stats, down_stats)


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
    cpu_loads = [x['cpu_load'].mean() for x in system_data]

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), cpu_loads, label='Average CPU load')
    autolabel(ax, rect)

    ax.set_ylabel('Load [%]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def plot_ram_usage(experiments: Dict) -> None:
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]

    total_mem = psutil.virtual_memory().total
    ram_usage = [(total_mem - x['mem_avail']).mean() for x in system_data]
    ram_usage = [x / float(1024 * 1024 * 1024) for x in ram_usage]

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), ram_usage, label='Average RAM usage')
    autolabel(ax, rect)

    # ax.set_ylim([0, total_mem + 3])
    ax.axhline(y=total_mem / float(1024 * 1024 * 1024),
               color='red',
               label='Max. available memory')
    ax.set_ylabel('Usage [GiB]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


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
    with plt.style.context('ggplot'):
        experiments = {
            '1 Client'  : '1Client_Benchmark',
            '5 Clients' : '5Clients_Benchmark',
            '10 Clients': '10Clients_Benchmark'
        }

        plot_avg_times_frames(experiments, feedback=True)
        plot_avg_times_frames(experiments, feedback=False)

    # plot_avg_times_runsample(experiments)
    # # plot_avg_times_framesample(experiments)
    # plot_cpu_loads(experiments)
    # plot_ram_usage(experiments)
    # plot_time_dist(experiments)
