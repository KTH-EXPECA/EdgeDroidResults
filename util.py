import math
from typing import NamedTuple

import pandas as pd
from scipy import stats

SAMPLE_FACTOR = 5
MIN_SAMPLES = 500
CONFIDENCE = 0.95
Z_STAR = 1.96

Stats = NamedTuple('Stats', [('mean', float),
                             ('std', float),
                             ('conf_lower', float),
                             ('conf_upper', float)])

ExperimentTimes = NamedTuple('ExperimentTimes',
                             [('processing', Stats),
                              ('uplink', Stats),
                              ('downlink', Stats)])


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
