#!/usr/bin/env python3

import json
from multiprocessing.pool import Pool
from typing import Dict

import click
import pandas as pd
from scapy.all import *

from lego_timing import LEGOTCPdumpParser
from util import sample_frame_stats

from concurrent_logging import LOGGER

START_WINDOW = 10.0


def load_results(client_idx) -> Dict:
    filename = '{:02}_stats.json'.format(client_idx)
    with open(filename, 'r') as f:
        return json.load(f, encoding='utf-8')


def parse_all_clients_for_run(run_idx, num_clients) -> pd.DataFrame:
    os.chdir('run_{}'.format(run_idx + 1))
    LOGGER.info('Processing %d clients for run %d',
                num_clients, run_idx + 1)

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
    df = df.astype(dtype={'run_id': int})
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

    # print('Parsing stats for client {}'.format(client_idx))
    LOGGER.info('Parsing stats for client %d', client_idx)

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
        frame_id = frame['frame_id']
        client_send = frame['sent']
        feedback = frame['feedback']

        try:
            not_found = None
            try:
                server_recv = server_in[frame_id].pop(0) + server_offset
            except KeyError as error:
                LOGGER.warning(
                    'Frame %d was not found in the incoming frame dump',
                    frame_id)
                not_found = error
            try:
                server_send = server_out[frame_id].pop(0) + server_offset
            except KeyError as error:
                LOGGER.warning(
                    'Frame %d was not found in the outgoing frame dump',
                    frame_id)
                not_found = error

            if not_found:
                raise not_found
        except KeyError:
            LOGGER.warning('Skipping frame %d for client %d',
                           frame_id, client_idx)
            continue

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

        # raise e

    df = pd.DataFrame.from_dict(n_data)
    df = df.astype(dtype={'feedback' : bool,
                          'client_id': int,
                          'frame_id' : int})
    return df


def load_system_stats_for_run(run_idx):
    os.chdir('run_{}'.format(run_idx + 1))

    # print('Processing system stats for run {}'.format(run_idx))
    LOGGER.info('Processing system stats for run %d', run_idx + 1)

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


def get_run_status(client_id, run_id):
    os.chdir('run_{}'.format(run_id + 1))
    # print('Loading run results for client {}, run {}'.format(client_id,
    # run_id))

    LOGGER.info('Loading run results for client %d, run %d',
                client_id, run_id + 1)
    data = load_results(client_id)

    status = dict(
        client_id=client_id,
        run_id=run_id,
        start=data['run_results']['init'],
        end=data['run_results']['end'],
        success=data['run_results']['success']
    )
    os.chdir('..')
    return status


@click.group()
def cli():
    pass


def __sample_data(experiment_id):
    os.chdir(experiment_id)
    frame_data = pd.read_csv('total_frame_stats.csv')
    run_data = pd.read_csv('total_run_stats.csv')

    sampl_feedback = sample_frame_stats(frame_data, run_data, feedback=True)
    sampl_nofeedback = sample_frame_stats(frame_data, run_data, feedback=False)

    sampl_feedback = {k: v._asdict()
                      for k, v in sampl_feedback._asdict().items()}
    sampl_nofeedback = {k: v._asdict()
                        for k, v in sampl_nofeedback._asdict().items()}

    with open('sampled_time_stats_feedback.json', 'w') as f:
        json.dump(sampl_feedback, f)

    with open('sampled_time_stats_nofeedback.json', 'w') as f:
        json.dump(sampl_nofeedback, f)

    os.chdir('..')


@cli.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
def sample_data(experiment_id):
    __sample_data(experiment_id)


def __prepare_task_stats(experiment_id, n_clients, n_runs):
    os.chdir(experiment_id)

    combinations = []
    for c in range(n_clients):
        for r in range(n_runs):
            combinations.append((c, r))

    with Pool(min(6, n_runs)) as pool:
        data = pool.starmap(get_run_status, combinations)

    df = pd.DataFrame(data)
    df = df.astype(
        dtype={
            'client_id': int,
            'run_id'   : int,
            'start'    : float,
            'end'      : float,
            'success'  : bool
        }
    )

    df.to_csv('total_run_stats.csv')
    os.chdir('..')


@cli.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('n_clients', type=int)
@click.argument('n_runs', type=int)
def prepare_task_stats(experiment_id, n_clients, n_runs):
    __prepare_task_stats(experiment_id, n_clients, n_runs)


def __prepare_client_stats(experiment_id, n_clients,
                           n_runs, only_system_stats=False):
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

    os.chdir('..')


@cli.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('n_clients', type=int)
@click.argument('n_runs', type=int)
@click.option('--only_system_stats', type=bool, default=False,
              help='Only prepare system stats.')
def prepare_client_stats(experiment_id, n_clients, n_runs, only_system_stats):
    __prepare_client_stats(experiment_id, n_clients, n_runs, only_system_stats)


@cli.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('n_clients', type=int)
@click.argument('n_runs', type=int)
def process_all(experiment_id, n_clients, n_runs):
    __prepare_client_stats(experiment_id, n_clients, n_runs, False)
    __prepare_task_stats(experiment_id, n_clients, n_runs)
    __sample_data(experiment_id)


if __name__ == '__main__':
    cli()
