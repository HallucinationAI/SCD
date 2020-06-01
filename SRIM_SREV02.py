import os
import torch
from torch import distributions as D
import numpy as np
import matplotlib.pyplot as plt
from lib import Simulator
from lib import RatioEstimator, LossCriterion

# SRIM Librraries
from srim import Ion, Layer, Target, TRIM
from srim.output import Results

# Parser for EXYZ.txt- Collision output
def parse_EXYZ(dir):
    """ Parse directory with expected structure <dir>/<ion_symbol>/<int>/EXYZ.txt
    This functions will collect the position <x, y, z> of every ion and recoil collision.
    It will write to a numpy array the parsing. This file will be about 10% of the original
    """
    positions = []
    with open(os.path.join(dir, 'EXYZ.txt'), 'rb') as f:
        for line in f.readlines():
            line = line.decode('latin-1')
            if line.startswith('0'):
                tokens = line.split()
                positions.append([float(tokens[0]), # ION Number
                                  float(tokens[1])*1000, # ION Energy (KeV)
                                  float(tokens[2]), # Depth (X) - Angstrom
                                  float(tokens[3]), # Y - Angstrom
                                  float(tokens[4]), # Z - Angstrom
                                  float(tokens[5]), # Electronic (eV/A)
                                  float(tokens[6])]) # Energy Lost to Last Recoil (eV)
    np.save(os.path.join(dir, 'exyz.dat'), np.array(positions))
    return positions

# Update the Layer Energy
def update_Layer(Ed_enery_postrior):
    layer = Layer({
        'Ni': {
            'stoich': 1.0,  # Stoichiometry of element
            'E_d': Ed_enery_postrior,  # Displacement Energy [eV]
            'lattice': 0.0,
            'surface': 3.0
        }}, density=8.9, width=20000.0)
    # Construct a target of a single layer of Nickel
    target = Target([layer])
    return target

#*************************************************************************************
# Simulator Setting
# Construct a 3MeV Nickel ion
# TRIM.copy_output_files('c:/SRIM-Pro', 'c:/SRIM-Pro/SRIM Outputs')
# Specify the directory of SRIM.exe
srim_executable_directory = 'c://SRIM-Pro'
# ION Type and Energy and Mass
ion = Ion('Ni', energy=3.0e6)

# Construct a layer of nick 20um thick with a displacement energy of 30 eV
# Parameters to be optimized
E_d_prior = 30.0 # Displacement Energy [eV]
number_of_IONs = 50

# PRIOR for Parameter to be Optimized
prior = D.Uniform(E_d_prior-3, E_d_prior+3)

# Architecture definitions for Ratio Estimator - Neural Network
activation = torch.nn.ELU
layers = [64, 64, 64]
inputs_shape = (7,)
outputs_shape = (1,)

ratio_estimator = RatioEstimator(inputs_shape, outputs_shape, activation, layers).to('cuda')
ratio_estimator.train()

## NOW TRAIN JUST for some number of epochs
nepochs = 10
optimizer = torch.optim.Adam(ratio_estimator.parameters())

losses = []
loss_criterion = LossCriterion(ratio_estimator=ratio_estimator).cuda()

for i in range(nepochs):

    # Program Loop
    # Prior Binding Energy (Parameter)
    # Step 1: Simulator ->  We set the Lattice Binding Energy (Parameter) and generate n sample form ION Collisions
    # Step 2: We Train the Network with n Batch of Samples ( Batch means ION)
    # Step 3: We Change Lattice Binding Energy by Posterior Energy

    # IONs are not independent and they change the lattice structure
    # Divide the file and make collision pattern for each ION
    """
    1: Simulate SRIM get Tensors 
    2: Put this Tensor into tensor_list 
    3: For each items in the tensor_list train the model
        - create theta sensors : batch_size = tensor_list[0].shape[0]
        - repeat in the batch : thetas = torch.tensor(E_d_prior).repeat([batch_size,1]).cuda()
        - repeat xs : xs = tensor_list[0].cuda()
    """

    Theta = prior.sample() # Parameter to be optimized (Displacement Energy [eV])
    # Simulator with this Theta and obtain a list of Collision
    # Convert the List to Collision Tensor (Torch)

    # Construct a target of a single layer of Nickel
    # Update the Layer
    layer_posteriori = update_Layer(Theta)

    # TRIM Setting for Simulation
    # Initialize a TRIM calculation with given target and ion for 25 ions, quick calculation
    trim = TRIM(layer_posteriori,
                ion,
                number_ions=number_of_IONs,
                calculation=1,  # full cascade
                exyz=1,  # Generate the Collision File
                collisions=1)

    # Do Simulation
    results = trim.run(srim_executable_directory)

    # If all went successful you should have seen a TRIM window popup and run NN ions!
    # results is `srim.output.Results` and contains all output files parsed
    # for Later usage (Abstract Statistics)
    results_c = Results(srim_executable_directory)

    # Sparse Results and Read the Information of the
    """
    =====================================================================
    Ion       Energy     Depth (X)     Y           Z       Electronic   Energy Lost to
    Number    (keV)    (Angstrom)  (Angstrom)  (Angstrom)  Stop.(eV/A)  Last Recoil(eV)
    ------- ----------- ---------- ----------- ----------- -----------  ---------------
    """
    collision = parse_EXYZ('c://SRIM-Pro//SRIM Outputs')
    # ION Number, Current Energy, X, Y, Z , SE, SN
    # Idea 1: Train the Network for Every ION , we have about 900~1000 Collision to Stop
    # input size -> [1000(random number) x 7]

    # Change to Torch tensor
    collision_tensor = torch.tensor(collision)

    for j in range(1, number_of_IONs+1):
        xs = collision_tensor[collision_tensor[:, 0] == j].cuda()
        batch_size = xs.shape[0]
        thetas = torch.tensor(Theta).repeat([batch_size, 1]).cuda()

        # Train for Each ION
        optimizer.zero_grad()
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

theta_produce_obs = prior.sample().to('cuda')
print('Lets choose an unknown theta : ', theta_produce_obs)


# Observation from real Data (From pavel)
real_Ed = E_d_prior

"""


print('This theta produce an observed x : ', observation)

# Now come up with most likeli thetas by choosing some intervals
resolution = 100
inputs = torch.linspace(E_d_prior-5, E_d_prior+5, resolution).view(-1, 1)
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

plt.plot(inputs.squeeze(), log_posterior.exp().squeeze())
plt.axvline(observation, lw=2, color="C0")
plt.xlabel(r"$\theta$")
plt.ylabel("Posterior density")
plt.title("Approximate posterior")
plt.show()
"""