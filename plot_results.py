import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab

from util import *

# n_runs = 25

PLOT_DIM = (4.5, 3)
SEPARATE_LEGEND = True
PLOT_TITLES = False
# PLOT_DIM = (8, 6)
FEEDBACK_TIME_RANGE = (-200, 500)
NO_FEEDBACK_TIME_RANGE = (-40, 100)

FEEDBACK_BIN_RANGE = (200, 1200)
NO_FEEDBACK_BIN_RANGE = (10, 300)


# HIST_FEEDBACK_YRANGE = (0, 0.025)
# HIST_NO_FEEDBACK_YRANGE = (0, 0.12)


def autolabel(ax: plt.Axes, rects: List[plt.Rectangle],
              y_range: Tuple[float, float],
              bottom: bool = False,
              color: str = 'black') -> None:
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2.0

        if not bottom:
            y_pos = 0.2 * (max(*y_range) - min(*y_range))
            ax.text(x_pos, y_pos,
                    '{:02.2f}'.format(height),
                    ha='center', va='bottom', weight='bold',
                    rotation='vertical',
                    color=color)
        else:
            y_pos = rect.get_y()
            ax.text(x_pos, y_pos,
                    '{:02.2f}'.format(height),
                    ha='center', va='top', weight='bold',
                    rotation='vertical',
                    color=color)


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

    if SEPARATE_LEGEND:
        figlegend = pylab.figure(figsize=(3, 1))
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
    else:
        ax.legend(loc='upper right', ncol=2)

    ax.set_xscale("log")
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Density')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #          ncol=2, mode="expand", borderaxespad=0.)

    fig.set_size_inches(*PLOT_DIM)
    if feedback:
        # ax.set_ylim(*HIST_FEEDBACK_YRANGE)
        fig.savefig('proc_hist_feedback.pdf', bbox_inches='tight')
        if PLOT_TITLES:
            plt.title('Processing times for frames w/ feedback')
    else:
        # ax.set_ylim(*HIST_NO_FEEDBACK_YRANGE)
        fig.savefig('proc_hist_nofeedback.pdf', bbox_inches='tight')
        if PLOT_TITLES:
            plt.title('Processing times for frames w/o feedback')
    plt.show()


def plot_avg_times_frames(experiments: Dict, feedback: bool = False) -> None:
    root_dir = os.getcwd()

    stats = []

    for exp_dir in experiments.values():
        os.chdir(root_dir + '/' + exp_dir)
        filename = 'sampled_time_stats_feedback.json' \
            if feedback else 'sampled_time_stats_nofeedback.json'
        with open(filename, 'r') as f:
            sampled_data = json.load(f)
        os.chdir(root_dir)

        stats.append(sampled_data)

    processing_means = [s['processing']['mean'] for s in stats]
    processing_errors = [
        [s['processing']['mean'] - s['processing']['conf_lower']
         for s in stats],
        [s['processing']['conf_upper'] - s['processing']['mean']
         for s in stats]]

    uplink_means = [s['uplink']['mean'] for s in stats]
    uplink_errors = [
        [s['uplink']['mean'] - s['uplink']['conf_lower']
         for s in stats],
        [s['uplink']['conf_upper'] - s['uplink']['mean']
         for s in stats]]

    downlink_means = [s['downlink']['mean'] for s in stats]
    downlink_errors = [
        [s['downlink']['mean'] - s['downlink']['conf_lower']
         for s in stats],
        [s['downlink']['conf_upper'] - s['downlink']['mean']
         for s in stats]]

    bar_width = 0.3
    r1 = np.arange(len(experiments))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    errorbar_opts = dict(
        fmt='none',
        linestyle='none',
        ecolor='black',
        lw=3, alpha=1.0,
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
        list(map(lambda r: autolabel(ax, r, FEEDBACK_TIME_RANGE, bottom=True),
                 rects))
        # force eval
        ax.set_ylim(*FEEDBACK_TIME_RANGE)
    else:
        list(map(lambda r: autolabel(ax, r, NO_FEEDBACK_TIME_RANGE,
                                     bottom=True),
                 rects))
        ax.set_ylim(*NO_FEEDBACK_TIME_RANGE)

    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)

    if SEPARATE_LEGEND:
        dim_x, dim_y = PLOT_DIM
        figlegend = pylab.figure(figsize=(dim_x*2.0, .3))
        figlegend.legend((up_err, *rects),
                         (up_err.get_label(), *(r.get_label() for r in rects)),
                         loc='center', mode='expand', ncol=4)
        figlegend.tight_layout()
        figlegend.savefig('times_legend.pdf', transparent=True,
                          bbox_inches='tight', pad_inches=0)
        figlegend.show()
    else:
        ax.legend(loc='upper left', ncol=2)

    # Add xticks on the middle of the group bars
    # ax.set_xlabel('Number of clients', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(experiments))])
    ax.set_xticklabels(experiments.keys())

    y_ticks = [tick for tick in ax.get_yticks() if tick >= 0]
    ax.set_yticks(y_ticks)

    fig.set_size_inches(*PLOT_DIM)
    if feedback:
        fig.savefig('times_feedback.pdf', bbox_inches='tight')
        if PLOT_TITLES:
            plt.title('Time statistics for frames w/ feedback')
    else:
        fig.savefig('times_nofeedback.pdf', bbox_inches='tight')
        if PLOT_TITLES:
            plt.title('Time statistics for frames w/o feedback')
    plt.show()


def plot_cpu_loads(experiments: Dict) -> None:
    # system_data = [load_system_data_for_experiment(x)
    #                for x in experiments.values()]
    system_data_samples = []
    for exp_name, exp_dir in experiments.items():
        data = load_system_data_for_experiment(exp_dir)
        runs = data['run'].unique()
        samples = []
        for run in runs:
            df = data.loc[data['run'] == run]
            samples.append(df.sample(n=2 * SAMPLE_FACTOR))
        system_data_samples.append(pd.concat(samples))

    cpu_means = [x['cpu_load'].mean() for x in system_data_samples]
    cpu_stds = [x['cpu_load'].std() for x in system_data_samples]
    cpu_count = [x.shape[0] for x in system_data_samples]

    cpu_confs = [
        stats.norm.interval(
            CONFIDENCE,
            loc=mean,
            scale=std / math.sqrt(count)
        )
        for mean, std, count in zip(cpu_means, cpu_stds, cpu_count)
    ]

    err = [
        [mean - x[0] for mean, x in zip(cpu_means, cpu_confs)],
        [x[1] - mean for mean, x in zip(cpu_means, cpu_confs)]
    ]

    cpu_range = (0, 100)

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), cpu_means, label='Average CPU load')
    ax.errorbar(experiments.keys(), cpu_means, yerr=err,
                fmt='none',
                linestyle='none',
                ecolor='darkblue',
                lw=10, alpha=1.0,
                capsize=0, capthick=1, label='95% Confidence Interval')

    autolabel(ax, rect, cpu_range, bottom=True)

    ax.set_ylabel('Load [%]')
    ax.set_ylim(*cpu_range)
    ax.legend(loc='upper left')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.set_size_inches(*PLOT_DIM)
    fig.savefig('cpu_load.pdf', bbox_inches='tight')
    plt.show()


def plot_ram_usage(experiments: Dict) -> None:
    system_data_samples = []
    for exp_name, exp_dir in experiments.items():
        data = load_system_data_for_experiment(exp_dir)
        runs = data['run'].unique()
        samples = []
        for run in runs:
            df = data.loc[data['run'] == run]
            samples.append(df.sample(n=2 * SAMPLE_FACTOR))
        system_data_samples.append(pd.concat(samples))

    mem_means = [x['mem_avail'].mean() for x in system_data_samples]
    mem_stds = [x['mem_avail'].std() for x in system_data_samples]
    mem_count = [x.shape[0] for x in system_data_samples]

    mem_confs = [
        stats.norm.interval(
            CONFIDENCE,
            loc=mean,
            scale=std / math.sqrt(count)
        )
        for mean, std, count in zip(mem_means, mem_stds, mem_count)
    ]

    err = [
        [mean - x[0] for mean, x in zip(mem_means, mem_confs)],
        [x[1] - mean for mean, x in zip(mem_means, mem_confs)]
    ]

    # total_mem = psutil.virtual_memory().total
    conv_factor = float(1024 * 1024 * 1024)  # GiB <-> MiB
    total_mem = 32 * conv_factor
    mem_usage_means = [(total_mem - m) / conv_factor for m in mem_means]
    err = [
        list(map(lambda m: m / conv_factor, err[0])),
        list(map(lambda m: m / conv_factor, err[1]))
    ]
    total_mem = total_mem / conv_factor  # convert back to GiB

    ram_range = (0, total_mem + 3)

    fig, ax = plt.subplots()
    rect = ax.bar(experiments.keys(), mem_usage_means,
                  label='Average RAM usage',
                  color='darkblue')
    ax.errorbar(experiments.keys(), mem_usage_means, yerr=err,
                fmt='none',
                linestyle='none',
                ecolor='darkorange',
                lw=10, alpha=1.0,
                capsize=0, capthick=1, label='95% Confidence Interval')

    autolabel(ax, rect, ram_range, bottom=True, color='white')

    ax.set_ylim(*ram_range)
    ax.axhline(y=total_mem,
               color='red',
               label='Max. available memory')
    ax.set_ylabel('Usage [GiB]')
    ax.legend(loc="center left")

    fig.set_size_inches(*PLOT_DIM)
    fig.savefig('ram_usage.pdf', bbox_inches='tight')

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
    with plt.style.context('seaborn-paper'):
        plt.rcParams['font.size'] = 10 # font size
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

        experiments = {
            '1 Client\nOptimal'        : '1Client_100Runs',
            '5 Clients\nOptimal'       : '5Clients_100Runs',
            '10 Clients\nOptimal'      : '10Clients_100Runs',
            '10 Clients\nImpaired\nWiFi': '10Clients_100Runs_BadLink'  # ,
            # 'Impaired\nCPU' : '10Clients_100Runs_0.5CPU'
        }

        # experiments = {
        #     '1 Client'  : '1Client_100Runs',
        #     '5 Clients' : '5Clients_100Runs',
        #     '10 Clients': '10Clients_100Runs',
        #     '15 Clients': '15Clients_100Runs'
        # }

        # experiments = {
        #     'TCPDUMP': '1Client_10Runs_ArtificialLoad',
        #     'No TCPDUMP': '1Client_10Runs_NoTCPDUMP'
        # }

        # os.chdir('1Client_100Runs_BadLink')
        # frame_data = pd.read_csv('total_frame_stats.csv')
        # run_data = pd.read_csv('total_run_stats.csv')
        # os.chdir('..')

        # print_successful_runs(experiments)

        plot_avg_times_frames(experiments, feedback=True)
        plot_avg_times_frames(experiments, feedback=False)
        #plot_time_dist(experiments, feedback=True)
        #plot_time_dist(experiments, feedback=False)

        #plot_cpu_loads(experiments)
        #plot_ram_usage(experiments)
