import itertools
import json
import math
import operator
import os
from statistics import mean
from typing import Dict, List, Tuple, NamedTuple
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from matplotlib import pylab
from scipy import stats

# n_runs = 25
CONFIDENCE = 0.95
Z_STAR = 1.96
SAMPLE_FACTOR = 5
MIN_SAMPLES = 500

Stats = NamedTuple('Stats', [('mean', float),
                             ('std', float),
                             ('conf_lower', float),
                             ('conf_upper', float)])

ExperimentTimes = NamedTuple('ExperimentTimes',
                             [('processing', Stats),
                              ('uplink', Stats),
                              ('downlink', Stats)])

PLOT_DIM = (4, 3)
FEEDBACK_TIME_RANGE = (0, 600)
NO_FEEDBACK_TIME_RANGE = (0, 100)

FEEDBACK_BIN_RANGE = (200, 800)
NO_FEEDBACK_BIN_RANGE = (10, 200)


def autolabel(ax: plt.Axes, rects: List[plt.Rectangle],
              y_range: Tuple[float, float]) -> None:
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2.0
        y_pos = 0.2 * (max(*y_range) - min(*y_range))
        ax.text(x_pos, y_pos,
                '{:02.2f}'.format(height),
                ha='center', va='bottom', weight='bold',
                rotation='vertical')


def filter_runs(frame_data: pd.DataFrame,
                run_data: pd.DataFrame) -> pd.DataFrame:
    n_runs = run_data['run_id'].max() + 1
    n_clients = run_data['client_id'].max() + 1

    samples = []
    for run in range(n_runs):
        for client in range(n_clients):
            success = run_data.loc[run_data['run_id'] == run]
            success = success.loc[success['client_id'] == client]
            success = success.iloc[0]['success']

            if success:
                d = frame_data.loc[frame_data['run_id'] == run]
                d = d.loc[d['client_id'] == client]
                samples.append(d)

    return pd.concat(samples)


def plot_time_dist(experiments: Dict, feedback: bool) -> None:
    root_dir = os.getcwd()
    results = {}

    for exp_name, exp_dir in experiments.items():
        os.chdir(root_dir + '/' + exp_dir)
        data = pd.read_csv('total_frame_stats.csv')
        run_data = pd.read_csv('total_run_stats.csv')
        os.chdir(root_dir)

        data = calculate_derived_metrics(data, feedback)
        data = filter_runs(data, run_data)

        results[exp_name] = data

    # bin_min = min(map(
    #     operator.methodcaller('min'),
    #     map(
    #         operator.itemgetter('processing'),
    #         results.values()
    #     )
    # ))
    #
    # bin_max = max(map(
    #     operator.methodcaller('max'),
    #     map(
    #         operator.itemgetter('processing'),
    #         results.values()
    #     )
    # ))

    fig, ax = plt.subplots()
    # bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 30)

    if feedback:
        bins = np.logspace(np.log10(FEEDBACK_BIN_RANGE[0]),
                           np.log10(FEEDBACK_BIN_RANGE[1]),
                           30)
    else:
        bins = np.logspace(np.log10(NO_FEEDBACK_BIN_RANGE[0]),
                           np.log10(NO_FEEDBACK_BIN_RANGE[1]),
                           30)

    hists = []
    pdfs = []
    for exp_name, data in results.items():
        hists.append(ax.hist(data['processing'], bins,
                             label=exp_name,
                             # norm_hist=True
                             alpha=0.5,
                             density=True)[-1])

        shape, loc, scale = stats.lognorm.fit(data['processing'])
        pdf = stats.lognorm.pdf(bins, shape, loc, scale)
        pdfs.append(*ax.plot(bins, pdf,
                             label=exp_name + ' lognorm PDF'))

    figlegend = pylab.figure(figsize=(3, 0.8))
    plots = (*(h[0] for h in hists), *pdfs)
    labels = (
        *(exp_name for exp_name, _ in results.items()),
        *(exp_name + ' PDF' for exp_name, _ in results.items())
    )
    figlegend.legend(plots,
                     labels,
                     loc='center',
                     mode='expand',
                     ncol=2)
    figlegend.tight_layout()
    figlegend.savefig('proc_hist_legend.pdf', transparent=True,
                      bbox_inches='tight', pad_inches=0)
    figlegend.show()

    ax.set_xscale("log")
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Density')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #          ncol=2, mode="expand", borderaxespad=0.)

    fig.set_size_inches(*PLOT_DIM)
    if feedback:
        fig.savefig('proc_hist_feedback.pdf', bbox_inches='tight')
        plt.title('Processing times for frames w/ feedback')
    else:
        fig.savefig('proc_hist_nofeedback.pdf', bbox_inches='tight')
        plt.title('Processing times for frames w/o feedback')
    plt.show()


def plot_avg_times_frames(experiments: Dict, feedback: bool = False) -> None:
    root_dir = os.getcwd()

    stats = []

    for exp_dir in experiments.values():
        os.chdir(root_dir + '/' + exp_dir)
        frame_data = pd.read_csv('total_frame_stats.csv', index_col=0)
        run_data = pd.read_csv('total_run_stats.csv')
        os.chdir(root_dir)

        stats.append(sample_frame_stats(frame_data,
                                        run_data,
                                        feedback=feedback))

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
        fmt='none',
        linestyle='none',
        ecolor='darkorange',
        lw=4, alpha=1.0,
        capsize=0, capthick=1
    )

    fig, ax = plt.subplots()
    up_err = ax.errorbar(r1, uplink_means, yerr=uplink_errors,
                         **errorbar_opts, label='95% Confidence Interval')
    proc_err = ax.errorbar(r2, processing_means, yerr=processing_errors,
                           **errorbar_opts)
    down_err = ax.errorbar(r3, downlink_means, yerr=downlink_errors,
                           **errorbar_opts)

    up_bars = ax.bar(r1, uplink_means,
                     label='Average uplink time',
                     # yerr=uplink_errors,
                     width=bar_width,
                     edgecolor='white',
                     # error_kw=dict(errorbar_opts, label='95% Confidence
                     # Interval')
                     )
    proc_bars = ax.bar(r2, processing_means,
                       label='Average processing time',
                       # yerr=processing_errors,
                       width=bar_width,
                       edgecolor='white',
                       # error_kw=errorbar_opts
                       )
    down_bars = ax.bar(r3, downlink_means,
                       label='Average downlink time',
                       # yerr=downlink_errors,
                       width=bar_width,
                       edgecolor='white',
                       # error_kw=errorbar_opts
                       )

    rects = (up_bars, proc_bars, down_bars)
    # autolabel(ax, rect1)
    # autolabel(ax, rect2)
    # autolabel(ax, rect3)

    ax.set_ylabel('Time [ms]')

    if feedback:
        list(map(lambda r: autolabel(ax, r, FEEDBACK_TIME_RANGE), rects))
        # force eval
        ax.set_ylim(*FEEDBACK_TIME_RANGE)
    else:
        list(map(lambda r: autolabel(ax, r, NO_FEEDBACK_TIME_RANGE), rects))
        ax.set_ylim(*NO_FEEDBACK_TIME_RANGE)

    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)

    figlegend = pylab.figure(figsize=(3, 1))
    figlegend.legend((up_err, *rects),
                     (up_err.get_label(), *(r.get_label() for r in rects)),
                     loc='center', mode='expand')
    figlegend.tight_layout()
    figlegend.savefig('times_legend.pdf', transparent=True,
                      bbox_inches='tight', pad_inches=0)
    figlegend.show()

    # Add xticks on the middle of the group bars
    # ax.set_xlabel('Number of clients', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(experiments))])
    ax.set_xticklabels(experiments.keys())

    fig.set_size_inches(*PLOT_DIM)
    if feedback:
        fig.savefig('times_feedback.pdf', bbox_inches='tight')
        plt.title('Time statistics for frames w/ feedback')
    else:
        fig.savefig('times_nofeedback.pdf', bbox_inches='tight')
        plt.title('Time statistics for frames w/o feedback')
    plt.show()


def sample_frame_stats(f_data: pd.DataFrame,
                       r_data: pd.DataFrame,
                       feedback: bool = False) -> ExperimentTimes:
    frame_data = calculate_derived_metrics(f_data, feedback)
    frame_data = filter_runs(frame_data, r_data)

    u_runs = frame_data['run_id'].unique()
    adj_sampl_factor = SAMPLE_FACTOR

    if feedback:
        samples = [frame_data.loc[frame_data['run_id'] == run_id].sample()
                   for run_id in u_runs]
    else:
        # find number of clients
        n_clients = frame_data['client_id'].max() + 1

        while True:
            # take SAMPLE_FACTOR samples per client per run
            samples = []
            for run_id in u_runs:
                run_data = frame_data.loc[frame_data['run_id'] == run_id]
                for client_id in range(n_clients):
                    client_data = run_data.loc[
                        run_data['client_id'] == client_id]

                    if not client_data.empty:
                        if adj_sampl_factor <= client_data.shape[0]:
                            samples.append(
                                client_data.sample(n=adj_sampl_factor)
                            )
                        else:
                            samples.append(client_data)

            if sum(map(lambda s: s.shape[0], samples)) > MIN_SAMPLES:
                break
            else:
                adj_sampl_factor += SAMPLE_FACTOR

        # if client_data.shape[0] >= SAMPLE_FACTOR:
        #     samples.append(client_data.sample(n=SAMPLE_FACTOR))
        # else:
        #    samples.append(client_data)

    samples = pd.concat(samples)
    print('Total samples:', samples.shape[0])
    print('Samples per successful run:', adj_sampl_factor)

    # stats for processing times:
    proc_mean = samples['processing'].mean()
    proc_std = samples['processing'].std()
    proc_conf = stats.norm.interval(
        CONFIDENCE,
        loc=proc_mean,
        scale=proc_std / math.sqrt(samples.shape[0])
    )

    proc_stats = Stats(proc_mean, proc_std, *proc_conf)

    # stats for uplink times:
    up_mean = samples['uplink'].mean()
    up_std = samples['uplink'].std()
    up_conf = stats.norm.interval(
        CONFIDENCE,
        loc=up_mean,
        scale=up_std / math.sqrt(samples.shape[0])
    )
    up_stats = Stats(up_mean, up_std, *up_conf)

    # stats for downlink times:
    down_mean = samples['downlink'].mean()
    down_std = samples['downlink'].std()
    down_conf = stats.norm.interval(
        CONFIDENCE,
        loc=down_mean,
        scale=down_std / math.sqrt(samples.shape[0])
    )
    down_stats = Stats(down_mean, down_std, *down_conf)

    return ExperimentTimes(proc_stats, up_stats, down_stats)


def calculate_derived_metrics(data, feedback):
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
    return frame_data


def plot_cpu_loads(experiments: Dict) -> None:
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]
    cpu_loads = [x['cpu_load'].mean() for x in system_data]

    cpu_range = (0, 100)

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), cpu_loads, label='Average CPU load')
    autolabel(ax, rect, cpu_range)

    ax.set_ylabel('Load [%]')
    ax.set_ylim(*cpu_range)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def plot_ram_usage(experiments: Dict) -> None:
    system_data = [load_system_data_for_experiment(x)
                   for x in experiments.values()]

    total_mem = psutil.virtual_memory().total
    ram_usage = [(total_mem - x['mem_avail']).mean() for x in system_data]
    ram_usage = [x / float(1024 * 1024 * 1024) for x in ram_usage]

    ram_range = (0, total_mem + 3)

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), ram_usage, label='Average RAM usage')
    autolabel(ax, rect, ram_range)

    ax.set_ylim(*ram_range)
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


def print_successful_runs(experiments):
    for exp_name, exp_id in experiments.items():
        os.chdir(exp_id)
        df = pd.read_csv('total_run_stats.csv')
        os.chdir('..')

        print(exp_name)
        n_clients = df['client_id'].max() + 1
        total_runs = df['run_id'].max() + 1
        for c in range(n_clients):
            client_runs = df.loc[df['client_id'] == c]
            success_runs = client_runs.loc[client_runs['success']].shape[0]
            print('Client {}: \t {} out of {} runs'
                  .format(c, success_runs, total_runs))


if __name__ == '__main__':
    with plt.style.context('ggplot'):
        experiments = {
            '1 Client'  : '1Client_100Runs_0.5CPU',
            '5 Clients' : '5Clients_100Runs_0.5CPU',
            '10 Clients': '10Clients_100Runs_0.5CPU'
        }

        os.chdir('1Client_100Runs_BadLink')
        frame_data = pd.read_csv('total_frame_stats.csv')
        run_data = pd.read_csv('total_run_stats.csv')
        os.chdir('..')

        print_successful_runs(experiments)

        plot_avg_times_frames(experiments, feedback=True)
        plot_avg_times_frames(experiments, feedback=False)
        plot_time_dist(experiments, feedback=True)
        plot_time_dist(experiments, feedback=False)

    # plot_avg_times_runsample(experiments)
    # # plot_avg_times_framesample(experiments)
    # plot_cpu_loads(experiments)
    # plot_ram_usage(experiments)
