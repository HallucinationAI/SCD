import torch
from torch import distributions as D
import numpy as np
import matplotlib.pyplot as plt
from lib import Simulator
from lib import RatioEstimator, LossCriterion


class NormalSimulator(Simulator):

    def __init__(self):
        super(NormalSimulator, self).__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 1)
        return torch.randn(inputs.size(0), 1).cuda() + inputs


simulator = NormalSimulator().to('cuda')

# PRIOR
prior = D.Uniform(-30, 30)

# Architecture definitions for Ratio Estimator
activation = torch.nn.ELU
layers = [64, 64, 64]
inputs_shape = (1,)
outputs_shape = (1,)

ratio_estimator = RatioEstimator(inputs_shape, outputs_shape, activation, layers).to('cuda')
ratio_estimator.train()

#### TRAIN the NETWORK
batch_size = 1024
thetas = prior.sample([batch_size, 1]).cuda()
xs = simulator(thetas)

##

loss_criterion = LossCriterion(ratio_estimator=ratio_estimator).cuda()
loss = loss_criterion(inputs=thetas, outputs=xs)

## NOW TRAIN JUST for some number of epochs
nepochs = 1000
optimizer = torch.optim.Adam(ratio_estimator.parameters())

losses = []
for i in range(nepochs):
    optimizer.zero_grad()
    thetas = prior.sample([batch_size, 1]).cuda()
    xs = simulator(thetas)

    loss = loss_criterion(inputs=thetas, outputs=xs)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

losses = np.array(losses)
plt.plot(np.log(losses), lw=2, color="black")
plt.minorticks_on()
plt.xlabel("Gradient updates")
plt.ylabel("Logarithmic loss")

plt.show()

## Posterior Estimate
a = 1

theta_produce_obs = prior.sample().to('cuda')
print('Lets choose an unknown theta : ', theta_produce_obs)

observation = simulator(theta_produce_obs)
print('This theta produce an observed x : ', observation)

# Now come up with most likeli thetas by choosing some intervals
resolution = 1000
inputs = torch.linspace(-30, 30, resolution).view(-1, 1)
outputs = observation.repeat(1, resolution).view(-1, 1)

## Now Compare likelihood ratios
'''
    r(theta, x) = P(theta|x) / P(theta) = P(x|theta) / P(x)
    P(theta|x) = r(x, theta) * P(theta)
     
'''

ratio_estimator.eval()

with torch.no_grad():
    log_posterior = ratio_estimator.log_ratio(inputs.to('cuda'), outputs).view(-1).to('cpu') + \
                    prior.log_prob(observation.to('cpu'))

a = 1

plt.plot(inputs.squeeze(), log_posterior.exp().squeeze())
plt.axvline(observation, lw=2, color="C0")
plt.xlabel(r"$\theta$")
plt.ylabel("Posterior density")
plt.title("Approximate posterior")
plt.show()
