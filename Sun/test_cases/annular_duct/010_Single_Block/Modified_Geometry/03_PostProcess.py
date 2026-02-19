from sun.src.sun_post_process import PostProcessSun
import matplotlib.pyplot as plt
import pickle


pp3010 = PostProcessSun('Output/results_20_40.pkl')
pp3010.extract_eigenfields()
pp3010.plot_eigenfields(save_filename='results_20_40')

omegar_an = [13450, 21077, 26721, 31296, 35049]
omegai_an = [0, 0, 0, 0, 0]

plt.figure()
plt.plot(omegar_an, omegai_an, 'x', mfc='none', ms=10, label='Analytical')
plt.plot(pp3010.data['Eigenfrequencies'].real, pp3010.data['Eigenfrequencies'].imag, 'o', mfc='none', ms=10, label=r'$30 \times 10$')
plt.legend()
plt.grid(alpha=0.2)
plt.xlim([12000, 36000])
plt.ylim([-5000, 5000])
plt.show()