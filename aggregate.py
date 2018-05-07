from process_results import *
import os
import numpy as np


def load_1Client_Experiment():
    os.chdir('1Client_IdealBenchmark')
    data = parse_client_stats(0)
    os.chdir('..')
    return data


def load_5Clients_Experiment():
    os.chdir('5Clients_IdealBenchmark')
    results = []
    for i in range(5):
        results.append(parse_client_stats(i))
    os.chdir('..')
    return results


def load_10Clients_Experiment():
    os.chdir('10Clients_IdealBenchmark')
    results = []
    for i in range(10):
        results.append(parse_client_stats(i))
    os.chdir('..')
    return results

def load_system_stats():
    df = pd.read_csv('system_stats.csv')
    start_time = df['timestamp'][0]
    df['timestamp'] = (df['timestamp'] - start_time) / (60 * 1000.0)
    df['avg_cpu_load'] = df['cpu_load'].rolling(11, center=True).mean()
    return df


def plot_avg_times():
    c1 = load_1Client_Experiment()
    c5 = load_5Clients_Experiment()
    c10 = load_10Clients_Experiment()

    avg_up = [c1['avg_up']]
    avg_proc = [c1['avg_proc']]
    avg_down = [c1['avg_down']]

    sum_up = 0
    count_up = 0
    sum_down = 0
    count_down = 0
    sum_proc = 0
    count_proc = 0

    for c in c5:
        sum_up += c['avg_up'] * c['count_up']
        sum_down += c['avg_down'] * c['count_down']
        sum_proc += c['avg_proc'] * c['count_proc']

        count_up += c['count_up']
        count_down += c['count_down']
        count_proc += c['count_proc']

    avg_up.append(sum_up / float(count_up))
    avg_down.append(sum_down / float(count_down))
    avg_proc.append(sum_proc / float(count_proc))

    sum_up = 0
    count_up = 0
    sum_down = 0
    count_down = 0
    sum_proc = 0
    count_proc = 0

    for c in c10:
        sum_up += c['avg_up'] * c['count_up']
        sum_down += c['avg_down'] * c['count_down']
        sum_proc += c['avg_proc'] * c['count_proc']

        count_up += c['count_up']
        count_down += c['count_down']
        count_proc += c['count_proc']

    avg_up.append(sum_up / float(count_up))
    avg_down.append(sum_down / float(count_down))
    avg_proc.append(sum_proc / float(count_proc))

    fig, ax = plt.subplots()

    x = ['1 client', '5 clients', '10 clients']

    rects1 = ax.bar(x, avg_up, label='Avg uplink time')
    rects2 = ax.bar(x, avg_proc, bottom=avg_up, label='Avg processing time')
    rects3 = ax.bar(x, avg_down,
                    bottom=[x + y for x, y in zip(avg_up, avg_proc)],
                    label='Avg downlink time')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)

    ax.set_ylabel('Time [ms]')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )
    plt.show()


def compare_cpu_loads():
    os.chdir('1Client_IdealBenchmark')
    c1_df = load_system_stats()
    os.chdir('..')

    os.chdir('5Clients_IdealBenchmark')
    c5_df = load_system_stats()
    os.chdir('..')

    os.chdir('10Clients_IdealBenchmark')
    c10_df = load_system_stats()
    os.chdir('..')

    fig, ax = plt.subplots()

    ax.plot(c1_df['timestamp'], c1_df['avg_cpu_load'], label='1 client')
    ax.plot(c5_df['timestamp'], c5_df['avg_cpu_load'], label='5 clients')
    ax.plot(c10_df['timestamp'], c10_df['avg_cpu_load'], label='10 clients')

    ax.set_ylabel('Load [%]')
    ax.set_xlabel('Time [m]')
    ax.set_ylim(0, 100)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('Average CPU load')
    plt.show()


if __name__ == '__main__':
    #plot_avg_times()
    compare_cpu_loads()
