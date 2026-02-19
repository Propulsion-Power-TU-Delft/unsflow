from sun.src.sun_post_process import PostProcessSun
import matplotlib.pyplot as plt
import pickle


pp3010 = PostProcessSun('Output/results_30_10.pkl')
pp3010.extract_eigenfields()
pp3010.plot_eigenfields()

with open('analytical/analytical_eigenvalues.pkl', 'rb') as f:  
    omega_analytical = pickle.load(f)

plt.figure()
plt.plot(omega_analytical.real, omega_analytical.imag, 'x', mfc='none', ms=10, label='Analytical')
plt.plot(pp3010.data['Eigenfrequencies'].real, pp3010.data['Eigenfrequencies'].imag, 'o', mfc='none', ms=10, label=r'$30 \times 10$')
plt.legend()
plt.grid(alpha=0.2)
plt.xlim([12000, 36000])
plt.ylim([-5000, 5000])
plt.show()