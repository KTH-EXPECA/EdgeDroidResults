#!/usr/bin/env python3

import json
from multiprocessing.pool import Pool

import click
import pandas as pd
from scapy.all import *

from lego_timing import LEGOTCPdumpParser

START_WINDOW = 10.0


def load_results(client_idx):
    filename = '{:02}_stats.json'.format(client_idx)
    with open(filename, 'r') as f:
        return json.load(f, encoding='utf-8')


def parse_all_clients_for_run(run_idx, num_clients):
    os.chdir('run_{}'.format(run_idx + 1))
    print('Processing {} clients for run {}'.format(num_clients, run_idx + 1))

    parser = LEGOTCPdumpParser('tcp.pcap')
    with open('server_stats.json', 'r') as f:
        server_stats = json.load(f)

    # client_ntp_offset = data['run_results']['ntp_offset']
    server_ntp_offset = server_stats['server_offset']
    run_start = server_stats['run_start']
    run_end = server_stats['run_end']

    start_cutoff = START_WINDOW * 1000.0 + run_start
    end_cutoff = run_end - START_WINDOW * 1000.0
    # with Pool(3) as pool:
    client_dfs = list(itertools.starmap(
        _parse_client_stats_for_run,
        zip(
            range(num_clients),
            itertools.repeat(parser),
            itertools.repeat(server_ntp_offset),
            itertools.repeat(start_cutoff),
            itertools.repeat(end_cutoff)
        )
    ))

    for cdf in client_dfs:
        cdf['run_id'] = run_idx

    df = pd.concat(client_dfs, ignore_index=True)
    os.chdir('..')
    return df

    # for i in range(num_clients):
    #     client_df = _parse_client_stats_for_run(i, parser, server_ntp_offset,
    #                                             start_cutoff, end_cutoff)
    #     client_df['run_id'] = run_idx
    #
    #     if df.empty:
    #         df = client_df
    #     else:
    #         df = pd.concat([df, client_df], ignore_index=True)
    #
    # os.chdir('..')
    # return df


def _parse_client_stats_for_run(client_idx, parser, server_offset,
                                start_cutoff, end_cutoff):
    data = load_results(client_idx)

    print('Parsing stats for client {}'.format(client_idx))

    video_port = data['ports']['video']
    result_port = data['ports']['result']

    server_in = parser.extract_incoming_timestamps(video_port)
    server_out = parser.extract_outgoing_timestamps(result_port)

    n_data = {
        'client_id'  : [],
        'frame_id'   : [],
        'feedback'   : [],
        'client_send': [],
        'server_recv': [],
        'server_send': [],
        'client_recv': []
    }

    for frame in data['run_results']['frames']:
        try:
            frame_id = frame['frame_id']
            client_send = frame['sent']
            feedback = frame['feedback']
            server_recv = server_in[frame_id].pop(0) + server_offset
            server_send = server_out[frame_id].pop(0) + server_offset
            client_recv = frame['recv']

            if client_send < start_cutoff or client_recv > end_cutoff:
                continue

            n_data['client_id'].append(client_idx)
            n_data['frame_id'].append(frame_id)
            n_data['feedback'].append(feedback)
            n_data['client_send'].append(client_send)
            n_data['server_recv'].append(server_recv)
            n_data['server_send'].append(server_send)
            n_data['client_recv'].append(client_recv)

        except KeyError as e:
            print(e)
            print(os.getcwd())
            print('Client: ', client_idx)
            print('Ports: ', {'video': video_port, 'result': result_port})
            raise e

    return pd.DataFrame.from_dict(n_data)


def load_system_stats_for_run(run_idx):
    os.chdir('run_{}'.format(run_idx + 1))

    print('Processing system stats for run {}'.format(run_idx))

    df = pd.read_csv('system_stats.csv')

    with open('server_stats.json', 'r') as f:
        server_stats = json.load(f)

    run_start = server_stats['run_start']
    run_end = server_stats['run_end']

    start_cutoff = START_WINDOW * 1000.0 + run_start
    end_cutoff = run_end - START_WINDOW * 1000.0

    df['run'] = run_idx
    df = df.loc[df['timestamp'] > start_cutoff]
    df = df.loc[df['timestamp'] < end_cutoff]
    # df['run_start_cutoff'] = start_cutoff
    # df['run_end_cutoff'] = end_cutoff

    os.chdir('..')

    return df


@click.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('n_clients', type=int)
@click.argument('n_runs', type=int)
@click.option('--only_system_stats', type=bool, default=False,
              help='Only prepare system stats.')
def prepare_client_stats(experiment_id, n_clients, n_runs, only_system_stats):
    os.chdir(experiment_id)

    with Pool(min(6, n_runs)) as pool:
        if not only_system_stats:
            runs_df = pool.starmap(
                parse_all_clients_for_run,
                zip(
                    range(n_runs),
                    itertools.repeat(n_clients)
                )
            )
            runs = pd.concat(runs_df, ignore_index=True)
            runs.to_csv('total_frame_stats.csv')

        system_dfs = pool.map(load_system_stats_for_run, range(n_runs))
        system_stats = pd.concat(system_dfs, ignore_index=True)
        system_stats.to_csv('total_system_stats.csv')

    # for run_idx in range(n_runs):
    #     if not only_system_stats:
    #         clients_df = parse_all_clients_for_run(run_idx, n_clients)
    #         if runs.empty:
    #             runs = clients_df
    #         else:
    #             runs = pd.concat([runs, clients_df], ignore_index=True)
    #
    #     system = load_system_stats_for_run(run_idx, n_clients)
    #     if system_stats.empty:
    #         system_stats = system
    #     else:
    #         system_stats = pd.concat([system_stats, system],
    # ignore_index=True)
    #
    # if not only_system_stats:
    #     runs.to_csv('total_frame_stats.csv')
    #
    # system_stats.to_csv('total_system_stats.csv')
    os.chdir('..')


if __name__ == '__main__':
    prepare_client_stats()
