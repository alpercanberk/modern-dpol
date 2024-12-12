
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


##LOGIT NORMAL
def logit(x):
    return torch.log(x) - torch.log(1-x)

def logit_normal_pdf(x, m, s):
    x = torch.tensor(x).clamp(1e-7, 1-1e-7)
    return (1/(s * np.sqrt(2*np.pi))) * (1/(x * (1-x))) * torch.exp(-(logit(x)-m)**2/(2*s**2))

def sample_logit_normal(bsz, num_train_timesteps, device, m=0.0, s=1.0, eps=1e-3):
    x = torch.linspace(0, 1, num_train_timesteps, device=device)
    prob = logit_normal_pdf(x, m=m, s=s) + eps
    prob = prob / prob.sum()
    return torch.multinomial(prob, bsz, replacement=True).long()

def mode_sampling_pdf(u, s):
    """
    Implements the mode sampling density function from equation (20)
    
    Args:
        u: Input values between 0 and 1
        s: Scale parameter
    Returns:
        Probability density at u
    """
    return 1 - u - s * (torch.cos(math.pi/2 * u)**2 - 1 + u)

def sample_mode_distribution(bsz, num_train_timesteps, device, s=1.0, eps=1e-3):
    """
    Sample from the mode distribution
    
    Args:
        bsz: Batch size
        num_train_timesteps: Number of timesteps
        device: Device to put tensors on
        s: Scale parameter
    Returns:
        Sampled timesteps
    """
    x = torch.linspace(0, 1, num_train_timesteps, device=device)
    prob = mode_sampling_pdf(x, s) + 1e-3  # Add small constant for numerical stability
    prob = prob / prob.sum()  # Normalize to get proper probability distribution
    return torch.multinomial(prob, bsz, replacement=True).long()


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1024,
        prediction_type: str = 'flow',
        sampling_weight: str = 'logit_normal',
    ):
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.sampling_weight = sampling_weight

        self.timesteps = None

    def add_noise(self, original_samples, noise, timesteps):

        timesteps = timesteps.to(original_samples.device).float()/self.num_train_timesteps

        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)
         
        return original_samples * timesteps + noise * (1 - timesteps)
    
    def sample_timesteps(self, bsz, device):
        if self.sampling_weight == 'logit_normal_centered':
            x = torch.linspace(0, 1, self.num_train_timesteps, device=device)
            prob = logit_normal_pdf(x, m=0.0, s=1.25) + 1e-3
            prob = prob / prob.sum()

            sample = torch.multinomial(prob, bsz, replacement=True).long()
            return sample
        elif self.sampling_weight == 'logit_normal_skewleft':
            x = torch.linspace(0, 1, self.num_train_timesteps, device=device)
            prob = logit_normal_pdf(x, m=-0.5, s=1.5) + 1e-3
            prob = prob / prob.sum()
            sample = torch.multinomial(prob, bsz, replacement=True).long()
            return sample
        elif self.sampling_weight == 'logit_normal_skewright':
            x = torch.linspace(0, 1, self.num_train_timesteps, device=device)
            prob = logit_normal_pdf(x, m=0.5, s=1.5) + 1e-3
            prob = prob / prob.sum()
            sample = torch.multinomial(prob, bsz, replacement=True).long()
            return sample
        else:
            return torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
    
    def set_timesteps(self, num_inference_steps):
        """
        Don't judge me, I just tried matching the Diffusion Policy inference API
        """
        self.timesteps = np.linspace(0, self.num_train_timesteps, num_inference_steps+1)[:-1]

    def step(self, model_output, timestep, sample, generator=None, **kwargs):
        
        dt = 1.0 / len(self.timesteps)
        sample = model_output * dt + sample
        
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=sample)

    def __len__(self):
        return self.config.num_train_timesteps
    
if __name__ == "__main__":
    scheduler = FlowMatchEulerDiscreteScheduler(1024)
    print(scheduler.add_noise(torch.randn(1, 1024), torch.tensor([1.,2.,3.,4.,5.]), noise=torch.randn(1, 1024)))    

    scheduler.set_timesteps(8)
    print(scheduler.add_noise(torch.randn(1, 1024), torch.tensor([1.,2.,3.,4.,5.]), noise=torch.randn(1, 1024)))

    scheduler.set_timesteps(16)
    print(scheduler.add_noise(torch.randn(1, 1024), torch.tensor([1.,2.,3.,4.,5.]), noise=torch.randn(1, 1024)))

    #do a step
    print(scheduler.step(torch.randn(1, 1024), torch.tensor([1.,2.,3.,4.,5.]), torch.randn(1, 1024)))