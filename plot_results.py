import psutil
import matplotlib.pyplot as plt
import pandas as pd

def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2.0
        y_pos = 0.5 * height + rect.get_y() if height >= 5.0 else \
            rect.get_y()
        ax.text(x_pos, y_pos,
                '{:02.2f}'.format(height),
                ha='center', va='bottom', weight='bold')

def plot_rtts(data):
    frames = [run['frames'] for run in data['runs']]
    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    x_lim = 0
    for i, run in enumerate(frames):
        rtts = [frame.rtt for frame in run]
        proc = [frame.processing for frame in run]

        idx = [frame.id for frame in run]
        ax1.scatter(idx, rtts, label='Run {}'.format(i + 1), s=1)
        ax2.scatter(idx, proc, label='Run {}'.format(i + 1), s=1)

        x_lim = max(x_lim, len(idx))

    # y_lim = max(ax1.get_ylim()[-1], ax2.get_ylim()[-1])

    ax1.set_xlim(-50, x_lim + 50)
    ax2.set_xlim(-50, x_lim + 50)
    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('Total RTT [ms]')
    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('Processing time [ms]')
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    plt.title('Client {}'.format(data['client_id']))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.savefig(
    #     'client_{}_rtts_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )
    plt.show()


def plot_avg_times(data):
    results = [(r['avg_up'], r['avg_down'], r['avg_proc'])
               for r in data['runs']]
    x = ['Run {}'.format(i) for i in range(len(data['runs']))]

    up = [y[0] for y in results]
    down = [y[1] for y in results]
    proc = [y[2] for y in results]

    fig, ax = plt.subplots()

    rects1 = ax.bar(x, up, label='Avg uplink time')
    rects2 = ax.bar(x, proc, bottom=up, label='Avg processing time')
    rects3 = ax.bar(x, down,
                    bottom=[x + y for x, y in zip(up, proc)],
                    label='Avg downlink time')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)

    ax.set_ylabel('Time [ms]')
    plt.title('Client {}'.format(data['client_id']))
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.savefig(
    #     'client_{}_avgtimes_.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )
    plt.show()


def plot_task_times(data):
    fig, ax = plt.subplots()
    task_times = [
        run['end'] - run['init'] for run in data['runs']
    ]
    x = ['Run {}'.format(i + 1) for i in range(len(data['runs']))]
    ax.bar(x, task_times)
    ax.set_ylabel('Total task time [ms]')
    plt.show()


def plot_task_times_from_frames(data):
    task_times = []
    x = []
    for i, run in enumerate(data['runs']):
        send_times = [frame.client_send for frame in run['frames']]
        recv_times = [frame.client_recv for frame in run['frames']]

        task_times.append(max(recv_times) - min(send_times))
        x.append('Run {}'.format(i + 1))

    fig, ax = plt.subplots()
    task_times = [t / 1000.0 for t in task_times]
    ax.bar(x, task_times)
    ax.set_ylabel('Total task time [s]')
    plt.show()


def plot_cpu_load():
    df = pd.read_csv('system_stats.csv')
    start_time = df['timestamp'][0]
    df['timestamp'] = (df['timestamp'] - start_time) / (60 * 1000.0)
    df['avg_cpu_load'] = df['cpu_load'].rolling(11, center=True).mean()

    ax = df.plot(x='timestamp', y='cpu_load', alpha=0.5, label='CPU Load')
    df.plot(x='timestamp', y='avg_cpu_load',
            ax=ax, color='red', label='Avg. CPU Load')

    ax.set_xlabel('Time [m]')
    ax.set_ylabel('Load [%]')
    # plt.savefig(
    #     'cpu_load.png',
    #     bbox_inches='tight'
    # )

    plt.show()


def plot_ram_usage():
    df = pd.read_csv('system_stats.csv')
    start_time = df['timestamp'][0]
    df['timestamp'] = (df['timestamp'] - start_time) / (60 * 1000.0)
    df['mem_avail'] = df['mem_avail'] / (1024.0 * 1024.0 * 1024)

    total_ram = psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024)
    df['used_mem'] = total_ram - df['mem_avail']

    ax = df.plot(x='timestamp', y='used_mem', color='orange', label='Used RAM')
    ax.axhline(y=total_ram, color='red', label='Total RAM')
    # ax.set_yscale('log')
    ax.set_xlabel('Time [m]')
    ax.set_ylabel('Memory [GiB]')
    plt.legend()
    # plt.savefig(
    #     'ram_usage.png'.format(data['client_id']),
    #     bbox_inches='tight'
    # )
    plt.show()
