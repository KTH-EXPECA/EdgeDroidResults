#!/usr/bin/env python3

import json
from collections import namedtuple

import pandas as pd
from scapy.all import *

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
    avg_up = 0
    count_up = 0
    avg_down = 0
    count_down = 0
    avg_proc = 0
    count_proc = 0

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
                avg_up += uplink
                count_up += 1

            if downlink >= 0:
                avg_down += downlink
                count_down += 1

            avg_proc += processing
            count_proc += 1
        except AssertionError as e:
            frames.append(Frame(frame_id, rtt, None, None, None,
                                client_send, client_recv,
                                server_send, server_recv))

    frames.sort(key=lambda x: x['id'])

    avg_up = avg_up / float(count_up)
    avg_down = avg_down / float(count_down)
    avg_proc = avg_proc / float(count_proc)

    data['run_results']['frames'] = frames
    data['run_results']['avg_up'] = avg_up
    data['run_results']['avg_down'] = avg_down
    data['run_results']['avg_proc'] = avg_proc

    data['run_results']['count_up'] = count_up
    data['run_results']['count_down'] = count_down
    data['run_results']['count_proc'] = count_proc

    return data


def load_system_stats_for_run(run_idx):
    os.chdir('run_{}'.format(run_idx + 1))
    df = pd.read_csv('system_stats.csv')
    df['run'] = run_idx
    os.chdir('..')

    return df


@click.command()
@click.argument('experiment_id',
                type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('n_clients', type=int)
@click.argument('n_runs', type=int)
def prepare_client_stats(experiment_id, n_clients, n_runs):
    os.chdir(experiment_id)
    runs = dict()
    system_stats = pd.DataFrame()
    for run_idx in range(n_runs):
        clients = parse_all_clients_for_run(n_clients, run_idx)
        runs['run_{}'.format(run_idx)] = clients

        system = load_system_stats_for_run(run_idx)
        if system_stats.empty:
            system_stats = system
        else:
            system_stats = pd.concat([system_stats, system], ignore_index=True)

    with open('total_stats.json', 'w') as f:
        json.dump(runs, f)

    system_stats.to_csv('total_system_stats.csv')

    os.chdir('..')


if __name__ == '__main__':
    prepare_client_stats()
