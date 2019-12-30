#!/usr/bin/env python3
"""
 Copyright 2019 Manuel Olguín
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import json
from json import JSONDecodeError
from typing import Dict
from scapy.all import *
from concurrent_logging import LOGGER


class LEGOTCPdumpParser():
    pkts = []

    def __init__(self, pcapf):
        self.pkts = rdpcap(pcapf)

    def extract_incoming_timestamps(self, dport: int) -> Dict[int, list]:
        # pkts = rdpcap(pcapf)
        processed_frames = dict()

        for pkt in self.pkts:
            if TCP in pkt and Raw in pkt and pkt[TCP].dport == dport:
                try:
                    # incoming timestamps are different
                    # we know the packet header only includes the {frame_id}
                    # part, so we can directly try to parse it using the info
                    # provided in payload itself

                    payload = bytes(pkt[TCP].payload)

                    (header_len,) = struct.unpack('>I', payload[:4])
                    (header,) = struct.unpack(
                        '>{}s'.format(header_len), payload[4:header_len + 4]
                    )

                    frame_id_json = json.loads(header.decode('utf-8'))
                    frame_id = frame_id_json['frame_id']
                    if frame_id not in processed_frames.keys():
                        processed_frames[frame_id] = list()

                    # store the time as milliseconds
                    processed_frames[frame_id].append(pkt.time * 1000.0)

                except UnicodeDecodeError:
                    LOGGER.warning('Incoming: Could not decode payload.')
                except JSONDecodeError:
                    LOGGER.warning('Incoming: Could not decode JSON string.')
                    LOGGER.warning('Header: %s', header.decode('utf-8'))
                except struct.error:
                    continue
                except Exception as error:
                    LOGGER.error('Incoming: Unhandled exception!')
                    LOGGER.error(error)
                    raise error

        return processed_frames

    def extract_outgoing_timestamps(self, sport: int) -> Dict[int, list]:
        # pkts = rdpcap(pcapf)
        processed_frames = dict()

        for pkt in self.pkts:
            if TCP in pkt and Raw in pkt and pkt[TCP].sport == sport:
                try:
                    # grab a slice of the payload.
                    # we skip the first 4 bytes, since that's reserved for the
                    # length of the message.
                    # the 80 bytes is a bit arbitrary, but it's just so we can
                    # be sure we grab enough data to get the "frame_id" part of
                    # the message
                    data = bytes(pkt[TCP].payload)[4:80].decode('utf-8')
                    idx = data.index('"frame_id"')
                    comma_idx = data.index(',', idx, -1)
                    frame_id_txt = data[idx:comma_idx]
                    frame_id_json = json.loads('{' + frame_id_txt + '}')

                    frame_id = frame_id_json['frame_id']
                    if frame_id not in processed_frames.keys():
                        processed_frames[frame_id] = list()

                    # store the time as milliseconds
                    processed_frames[frame_id].append(pkt.time * 1000.0)

                except UnicodeDecodeError:
                    LOGGER.warning('Outgoing: Could not decode payload.')
                except JSONDecodeError:
                    LOGGER.warning('Outgoing: Could not decode JSON string.')
                except ValueError:
                    # LOGGER.warning('Outgoing: Could not find the frame
                    # header string.')
                    continue
                except Exception as error:
                    LOGGER.error('Outgoing: Unhandled exception!')
                    LOGGER.error(error)
                    raise error

        return processed_frames


if __name__ == '__main__':
    parser = LEGOTCPdumpParser('10Clients_TestBenchmark/run_1/tcp.pcap')
    print(parser.extract_incoming_timestamps(60018))
    print(parser.extract_outgoing_timestamps(60019))
