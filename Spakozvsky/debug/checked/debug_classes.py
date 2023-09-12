import sys
import Spakozvsky

from Spakozvsky.src.axial_gap import AxialGap
prova = AxialGap(0, 0.3, 1, 0.1)
M = prova.transfer_function(0, 1-1j, 2)
print(M)


from Spakozvsky.src.axial_rotor import AxialRotor
prova = AxialRotor(0.4, 0.8, 0.2, 0.3, 0.4, 0.3, 0.8, 0.9, 0.5)
M = prova.transfer_function(0, 1, 1)
print(M)


from Spakozvsky.src.axial_stator import AxialStator
prova = AxialStator(0.3, 0.1, 0.8, 0.2, 0.4, 0.4, 0.3, 1)
M = prova.transfer_function(0, 1, 1)
print(M)


from Spakozvsky.src.axial_duct import AxialDuct
prova = AxialDuct(0.3, 0.9)
M = prova.transfer_function(0.3, 0.1, 1, 1)
print(M)


from Spakozvsky.src.radial_impeller import RadialImpeller
prova = RadialImpeller(0.1, 0.4, 0.3, 0.4, 0.4, 0.5, 0.1, 0.1, 0.5, 0.4, 0.1, 0.3, 0.4, 0.8, -0.1, 0.1)
M = prova.transfer_function(0, 1-1j, 3)
print(M)


from Spakozvsky.src.swirling_flow import SwirlingFlow
prova = SwirlingFlow(0.1, 0.2, 0.3)
M = prova.transfer_function(0.4, 0, 1, 1)
print(M)


from Spakozvsky.src.vaned_diffuser import VanedDiffuser
prova = VanedDiffuser(0.3, 0.4, 0.9, 1.0, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.05, 0.9, 0.8, 0.8)
M = prova.transfer_function(0, 1, 1)
print(M)


from Spakozvsky.src.vaneless_diffuser import VanelessDiffuser
prova = VanelessDiffuser(0.3, 0.8, 0.3, 0.2)
M = prova.transfer_function(0.4, 0, 1, 1)
print(M)


