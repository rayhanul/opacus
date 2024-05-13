# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis

import numpy as np 

class DMAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self):
        super().__init__()
        self.previous_epsilons={}
    

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def M(self, k, theta, alpha):
        """
        return the moment generating function of a gamma distribution for t : (1-(x * theta))^(-k)
        """

        return np.power((1- np.multiply(alpha, theta)), -k)
    
    def get_epsilon_for_alpha(self, delta, k, theta, time, alpha):
        """
        This is inline funciton.
        it calculates frac{1}{alpha-1} log[  frac{alpha}{2 alpha-1} prod_{t=1}^T (1-(alpha-1)theta_t)^{-k_t} + frac{log (1/delta)}{alpha-1}
        """
        log_value=np.float128(0.0)
        log_value= np.multiply(alpha/((2 * alpha) -1), self.M(k, theta, alpha-1)) 
        if time >1:
            for _ in range(1, time):
                log_value = np.multiply(log_value, self.M( k, theta, (alpha-1)))
            log_value += (np.log((1/delta)))/(alpha-1)

        log_output=np.log(log_value)
        values =np.multiply((1/(alpha-1)) , log_output)
        return values

    def get_privacy_spent(self, *, delta: float,k=float, theta=float, time=int, alphas: Optional[List[Union[float, int]]] = None) -> Tuple[float, float]:
        if not self.history:
            return 0, 0

        if alphas is None:
            alphas = self.DEFAULT_ALPHAS

        valid_alphas= [alpha for alpha in alphas if alpha < (1/theta)]

        # print(f'number of steps: {self.history[0][2]}')
        # rdp = sum(
        #     [
        #         privacy_analysis.compute_rdp(
        #             q=sample_rate,
        #             noise_multiplier=noise_multiplier,
        #             steps=num_steps,
        #             orders=alphas,
        #         )
        #         for (noise_multiplier, sample_rate, num_steps) in self.history
        #     ]
        # )
        # eps, best_alpha = privacy_analysis.get_privacy_spent(
        #     orders=alphas, rdp=rdp, delta=delta
        # )
        # print(f'min eps:{eps}, and best alpha: {best_alpha}')

        epsilons= [ self.get_epsilon_for_alpha(delta=delta, k=k, theta=theta, time=time, alpha=alpha) for alpha in valid_alphas]

        # positive epsilon 
        positive_epsilons = [(index, value) for index, value in enumerate(epsilons) if value > 0 ]
        min_value_with_index = min(positive_epsilons, key=lambda item: item[1])

        print(f'time : {time} : {min_value_with_index[1] * len(self.history)}')
        if len(min_value_with_index) ==0: 
            return 0, 0
        return float(min_value_with_index[0]), float(min_value_with_index[1]) * len(self.history)



    def get_epsilon(self, delta: float, k:Optional[float]=None, theta:Optional[float]=None, time=Optional[int], alpha: Optional[List[Union[float, int]]] = None):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        eps, _ = self.get_privacy_spent(delta=delta, k=k, theta=theta, time=time)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "ddp"
