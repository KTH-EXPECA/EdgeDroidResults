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
import os

experiments = {
        '1 Client': '1Client_IdealBenchmark',
        '5 Clients': '5Clients_IdealBenchmark',
        '10 Clients': '10Clients_IdealBenchmark'
    }

for exp, dir in experiments.items():
    os.chdir(dir)

    for r in range(1, 6):
        os.chdir('run_{}'.format(r))

        with open('server_stats.json', 'r') as f:
            data = json.load(f)

        with open('server_stats.json', 'w') as f:
            data['run_start'] = data['run_start'] - 7200000
            data['run_end'] = data['run_end'] - 7200000
            json.dump(data, f)

        os.chdir('..')
    os.chdir('..')
