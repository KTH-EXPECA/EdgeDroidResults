#!/usr/bin/env python3

import json
from collections import namedtuple

import pandas as pd
from scapy.all import *
from statistics import mean, stdev
import click

from lego_timing import LEGOTCPdumpParser

STAGGER_INTERVAL = 1.0

Frame = namedtuple('Frame', ['id', 'rtt', 'uplink',
                             'downlink', 'processing',
                             'client_send', 'client_recv',
                             'server_send', 'server_recv'])


def load_results(client_idx):
    filename = '{:02}_stats.json'.format(client_idx)
    with open(filename, 'r') as f:
        return json.load(f, encoding='utf-8')


def parse_all_clients_for_run(num_clients, run_idx):
    os.chdir('run_{}'.format(run_idx + 1))
    parser = LEGOTCPdumpParser('tcp.pcap')
    with open('server_stats.json', 'r') as f:
        server_stats = json.load(f)

    # client_ntp_offset = data['run_results']['ntp_offset']
    server_ntp_offset = server_stats['server_offset']
    run_start = server_stats['run_start']
    run_end = server_stats['run_end']

    start_cutoff = num_clients * STAGGER_INTERVAL * 1000.0 + run_start
    end_cutoff = run_end - num_clients * STAGGER_INTERVAL * 1000.0

    clients = dict()
    for i in range(num_clients):
        clients['client_{}'.format(i)] = \
            _parse_client_stats_for_run(i, parser, start_cutoff,
                                        end_cutoff, server_ntp_offset)

    os.chdir('..')
    return clients


def _parse_client_stats_for_run(client_idx, parser,
                                start_cutoff, end_cutoff, server_offset):
    data = load_results(client_idx)
    video_port = data['ports']['video']
    result_port = data['ports']['result']

    # parser = LEGOTCPdumpParser('{:02}_dump.pcap'.format(client_idx))

    server_in = parser.extract_incoming_timestamps(video_port)
    server_out = parser.extract_outgoing_timestamps(result_port)

    frames = []
    up = []
    down = []
    proc = []

    for frame in data['run_results']['frames']:
        try:
            frame_id = frame['frame_id']
            client_send = frame['sent']
            server_recv = server_in[frame_id].pop(0) + server_offset
            server_send = server_out[frame_id].pop(0) + server_offset
            client_recv = frame['recv']
        except KeyError as e:
            print(e)
            print(os.getcwd())
            print('Client: ', client_idx)
            print('Ports: ', {'video': video_port, 'result': result_port})
            raise e

        if client_send < start_cutoff or client_recv > end_cutoff:
            continue

        uplink = server_recv - client_send
        processing = server_send - server_recv
        downlink = client_recv - server_send
        rtt = client_recv - client_send

        try:
            assert processing > 0
            frames.append(Frame(frame_id, rtt, uplink,
                                downlink, processing,
                                client_send, client_recv,
                                server_send, server_recv)._asdict())

            if uplink >= 0:
                up.append(uplink)

            if downlink >= 0:
                down.append(downlink)

            proc.append(processing)
        except AssertionError as e:
            frames.append(Frame(frame_id, rtt, None, None, None,
                                client_send, client_recv,
                                server_send, server_recv))

    frames.sort(key=lambda x: x['id'])

    data['run_results']['frames'] = frames
    data['run_results']['avg_up'] = mean(up)
    data['run_results']['avg_down'] = mean(down)
    data['run_results']['avg_proc'] = mean(proc)

    data['run_results']['std_up'] = stdev(up)
    data['run_results']['std_down'] = stdev(down)
    data['run_results']['std_proc'] = stdev(proc)

    data['run_results']['count_up'] = len(up)
    data['run_results']['count_down'] = len(down)
    data['run_results']['count_proc'] = len(proc)

    return data


def load_system_stats_for_run(run_idx, num_clients):
    os.chdir('run_{}'.format(run_idx + 1))
    df = pd.read_csv('system_stats.csv')

    with open('server_stats.json', 'r') as f:
        server_stats = json.load(f)

    run_start = server_stats['run_start']
    run_end = server_stats['run_end']

    start_cutoff = num_clients * STAGGER_INTERVAL * 1000.0 + run_start
    end_cutoff = run_end - num_clients * STAGGER_INTERVAL * 1000.0

    df['run'] = run_idx
    df['run_start_cutoff'] = start_cutoff
    df['run_end_cutoff'] = end_cutoff

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
    runs = dict()
    system_stats = pd.DataFrame()
    for run_idx in range(n_runs):
        if not only_system_stats:
            clients = parse_all_clients_for_run(n_clients, run_idx)
            runs['run_{}'.format(run_idx)] = clients

        system = load_system_stats_for_run(run_idx, n_clients)
        if system_stats.empty:
            system_stats = system
        else:
            system_stats = pd.concat([system_stats, system], ignore_index=True)

    if not only_system_stats:
        with open('total_stats.json', 'w') as f:
            json.dump(runs, f)

    system_stats.to_csv('total_system_stats.csv')
    os.chdir('..')


if __name__ == '__main__':
    prepare_client_stats()
