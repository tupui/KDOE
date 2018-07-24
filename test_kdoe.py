from itertools import combinations_with_replacement
import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from plotly.offline import plot
from batman.space import Space
from doe_kde import KdeSampler


def discrepancy_2D(sample):
    """Mean discrepancy of all 2D subprojections."""
    sample = np.asarray(sample)
    dim = sample.shape[1]

    # disc = ot.SpaceFillingC2().evaluate(np.stack([sample[:, 20],
    #                                                sample[:, 8]], axis=-1))

    disc = []
    for i, j in combinations_with_replacement(range(dim), 2):
        if i < j:
            disc_ = ot.SpaceFillingC2().evaluate(np.stack([sample[:, i],
                                                           sample[:, j]], axis=-1))
            # disc_ = ot.SpaceFillingMinDist().evaluate(np.stack([sample[:, i],
            #                                                     sample[:, j]], axis=-1))
            # disc_ = Space.discrepancy(np.stack([sample[:, i],
            #                                     sample[:, j]], axis=-1), method='MD')

            disc.append(disc_)

    return np.mean(disc)


dim = 2
n_sample = 10
sigma = 0.5
sampler = KdeSampler(sample=[[0.5, 0.7]], dim=dim, bw=sigma)
sample_kde = sampler.generate(n_sample)

dists = [ot.Uniform(0, 1) for _ in range(dim)]
dists = ot.ComposedDistribution(dists)
lhs = ot.LHSExperiment(dists, n_sample)
lhs_opt = ot.SimulatedAnnealingLHS(lhs, ot.GeometricProfile(), ot.SpaceFillingC2())

sample_lhs = np.array(lhs.generate())
sample_lhs_opt = np.array(lhs_opt.generate())
sample_sobol = np.array(ot.SobolSequence(dim).generate(n_sample))

print(f'Discrepancy CD:\n'
      f'-> KDE: {ot.SpaceFillingC2().evaluate(sample_kde)}\n'
      f'-> LHS opt: {ot.SpaceFillingC2().evaluate(sample_lhs_opt)}\n'
      f'-> LHS: {ot.SpaceFillingC2().evaluate(sample_lhs)}\n'
      f'-> Sobol: {ot.SpaceFillingC2().evaluate(sample_sobol)}\n')

print(f'Discrepancy WD:\n'
      f"-> KDE: {Space.discrepancy(sample_kde, method='WD')}\n"
      f"-> LHS opt: {Space.discrepancy(sample_lhs_opt, method='WD')}\n"
      f"-> LHS: {Space.discrepancy(sample_lhs, method='WD')}\n"
      f"-> Sobol: {Space.discrepancy(sample_sobol, method='WD')}\n")

print(f'Discrepancy MD:\n'
      f"-> KDE: {Space.discrepancy(sample_kde, method='MD')}\n"
      f"-> LHS opt: {Space.discrepancy(sample_lhs_opt, method='MD')}\n"
      f"-> LHS: {Space.discrepancy(sample_lhs, method='MD')}\n"
      f"-> Sobol: {Space.discrepancy(sample_sobol, method='MD')}\n")

print(f'Discrepancy 2D subprojections:\n'
      f'-> KDE: {discrepancy_2D(sample_kde)}\n'
      f'-> LHS opt: {discrepancy_2D(sample_lhs_opt)}\n'
      f'-> LHS: {discrepancy_2D(sample_lhs)}\n'
      f'-> Sobol: {discrepancy_2D(sample_sobol)}\n')

# from batman.visualization import doe
# # doe(sample_kde, plabels=['$x_1$', '$x_2$'])
# # raise

# print(f'Discrepancy subprojections 20-8:\n'
#       f'-> KDE: {ot.SpaceFillingC2().evaluate(np.stack([sample_kde[:, 20], sample_kde[:, 8]], axis=-1))}\n'
#       f'-> LHS opt: {ot.SpaceFillingC2().evaluate(np.stack([sample_lhs_opt[:, 20], sample_lhs_opt[:, 8]], axis=-1))}\n'
#       f'-> LHS: {ot.SpaceFillingC2().evaluate(np.stack([sample_lhs[:, 20], sample_lhs[:, 8]], axis=-1))}\n'
#       f'-> Sobol: {ot.SpaceFillingC2().evaluate(np.stack([sample_sobol[:, 20], sample_sobol[:, 8]], axis=-1))}\n')

# doe(sample_kde)

# doe(np.stack([sample_kde[:, 20], sample_kde[:, 8]], axis=-1), plabels=[r"$x_{20}$", r"$x_8$"])
# doe(np.stack([sample_lhs[:, 20], sample_lhs[:, 8]], axis=-1), plabels=[r"$x_{20}$", r"$x_8$"])
# doe(np.stack([sample_sobol[:, 20], sample_sobol[:, 8]], axis=-1), plabels=[r"$x_{20}$", r"$x_8$"])
# doe(np.stack([sample_lhs_opt[:, 20], sample_lhs_opt[:, 8]], axis=-1), plabels=[r"$x_{20}$", r"$x_8$"])


########################### Visualization #####################################

################################ 2D ###########################################

mini, maxi = -0., 1.


# KDOE
xx, yy = np.mgrid[mini:maxi:200j, mini:maxi:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel = np.exp(sampler.kde_prev.score_samples(positions.T))
f = np.reshape(kernel, xx.shape)

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(mini, maxi)
ax.set_ylim(mini, maxi)

# Rescaling to see previous KDE to have same as sampler.pdf(positions.T)
sigma_fin = sigma / (len(sample_kde) - 1) ** (1 / dim)
f_ = 1 - (2 * np.pi) ** (dim / 2) * sigma_fin ** dim * f * (len(sample_kde) - 1)  # gaussian
# # f_ = 1 - np.pi * sigma_fin ** 2 * f * (len(sample_kde) - 1)  # tophat
f_[np.where(f_ < 0)] = 0

bounds = np.linspace(0, 1, 50, endpoint=True)
contour = ax.contourf(xx, yy, f_, bounds)

ax.scatter(np.array(sample_kde)[:-1, 0], np.array(sample_kde)[:-1, 1],
           c='k', marker='o')
ax.scatter(np.array(sampler.sample_)[:, 0], np.array(sampler.sample_)[:, 1],
           c='r', marker='s')
ax.scatter(np.array(sample_kde)[-1, 0], np.array(sample_kde)[-1, 1],
           c='b', marker='D', s=60)

ticks = np.linspace(0, 1, num=5)
plt.colorbar(contour, ticks=ticks)
plt.show()

# WD plot
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(mini, maxi)
ax.set_ylim(mini, maxi)
f_ = np.array([Space.discrepancy(np.vstack([np.array(sample_kde)[:-1], s]), method='WD')
               for s in positions.T])[:, None]
f_ = np.reshape(f_, xx.shape)
contour = ax.contourf(xx, yy, f_)
ax.scatter(np.array(sample_kde)[:-1, 0], np.array(sample_kde)[:-1, 1],
           c='k', marker='o')
plt.colorbar(contour)
plt.show()

# 1D line at y=0.5
# fig = plt.figure()
# x = np.linspace(0, 1, 100)[:, None]
# x = np.concatenate([x, np.ones((100, 1)) * 0.5], axis=1)
# f_ = np.array([Space.discrepancy(np.vstack([np.array(sample_kde)[:-1], s]), method='MD')
#                for s in x])[:, None]
# plt.plot(x[:, 0], f_)
# plt.show()

################################ 2D bis #######################################

# mini = 0
# maxi = 1
# X, Y = np.mgrid[mini:maxi:100j, mini:maxi:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(np.exp(sampler.kde_prev.score_samples(positions.T)), X.shape)
# fig, ax = plt.subplots()
# ax.scatter(np.array(sample_kde)[:-1, 0], np.array(sample_kde)[:-1, 1], c='k', marker='o')
# # ax.scatter(np.array(sampler.sample_)[:, 0], np.array(sampler.sample_)[:, 1], c='r', marker='s')
# # ax.scatter(np.array(sample_kde)[-1, 0], np.array(sample_kde)[-1, 1], c='b', marker='D', s=60)
# ax.set_xlim((mini, maxi))
# ax.set_ylim((mini, maxi))
# plt.imshow(np.rot90(Z), extent=[mini, maxi, mini, maxi], interpolation='lanczos')
# # plt.clim(0.006, 0.009)
# plt.colorbar()
# plt.show()

################################ 3D ###########################################

# mini, maxi = -0., 1.

# # Peform the kernel density estimate
# xx, yy, zz = np.mgrid[mini:maxi:55j, mini:maxi:55j, mini:maxi:55j]
# positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
# f = np.exp(sampler.kde_prev.score_samples(positions.T))

# order = np.where((0.0152 < f) & (f < 0.0159))[0]

# xx = xx.ravel()
# yy = yy.ravel()
# zz = zz.ravel()

# scatter = dict(
#     mode="markers",
#     name="scatter",
#     type="scatter3d",
#     x=xx[order], y=yy[order], z=zz[order],
#     marker=dict(size=2, color="rgb(23, 190, 207)")
# )
# clusters = dict(
#     alphahull=7,
#     name="clusters",
#     opacity=0.1,
#     type="mesh3d",
#     x=xx[order], y=yy[order], z=zz[order]
# )

# layout = dict(title='3d point clustering', showlegend=True)

# fig = dict(data=[scatter],#, clusters],
#            layout=layout)
# # Use py.iplot() for IPython notebook
# plot(fig, filename='3d point clustering')

# raise

################################ Animation ####################################

# mini, maxi = -0., 1.
# xx, yy = np.mgrid[mini:maxi:100j, mini:maxi:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])

# sample_kde = [[0.5, 0.7]]

# for n_s in range(3, 20):

#     sampler = KdeSampler(sample=sample_kde, dim=dim, bw=1/2 * len(sample_kde))
#     sample_kde = sampler.generate(n_s - len(sample_kde))
#     kernel = np.exp(sampler.kde_prev.score_samples(positions.T))
#     f = np.reshape(kernel, xx.shape)

#     fig = plt.figure()
#     ax = fig.gca()
#     ax.set_xlim(mini, maxi)
#     ax.set_ylim(mini, maxi)
#     contour = ax.contourf(xx, yy, f)
#     ax.scatter(np.array(sample_kde)[:-1, 0], np.array(sample_kde)[:-1, 1],
#                c='k', marker='o')
#     ax.scatter(np.array(sampler.sample_)[:, 0], np.array(sampler.sample_)[:, 1],
#                c='r', marker='s')
#     ax.scatter(np.array(sample_kde)[-1, 0], np.array(sample_kde)[-1, 1],
#                c='b', marker='D', s=60)
#     plt.colorbar(contour)
#     fig.tight_layout()
#     fig.savefig('kde_conv_' + str(n_s) + '.pdf', transparent=True, bbox_inches='tight')
