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

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .prv import PRVAccountant
from .rdp import RDPAccountant
from .ddp import DMAccountant

__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
]


def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "rdp":
        return RDPAccountant()
    elif mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "prv":
        return PRVAccountant()
    elif mechanism == "ddp":
        return DMAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")