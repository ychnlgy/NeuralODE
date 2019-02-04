import numpy

def generate_spiral2d(
    nspiral=1000,
    ntotal=500,
    nsample=100,
    start=0,
    stop=6*numpy.pi,  # approximately equal to 6pi
    noise_std=0.1,
    a=0,
    b=0.3
):
    """

    This script is copied and modified from:
    https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py

    Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    numpy.random.seed(1337)

    # add 1 all timestamps to avoid division by 0
    orig_ts = numpy.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1 - orig_ts
    rs_cw = a + b * 50.0 / zs_cw
    xs, ys = rs_cw * numpy.cos(zs_cw) - 5, rs_cw * numpy.sin(zs_cw)
    orig_traj_cw = numpy.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * numpy.cos(zs_cc) + 5, rw_cc * numpy.sin(zs_cc)
    orig_traj_cc = numpy.stack((xs, ys), axis=1)

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    print(nspiral)
    for _ in range(nspiral):
        
        # don't sample t0 very near the start or the end
        t0_idx = numpy.random.multinomial(
            1, [1. / (ntotal - 2 * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = numpy.argmax(t0_idx) + nsample

        cc = bool(numpy.random.rand() > 0.5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += numpy.random.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = numpy.stack(orig_trajs, axis=0)
    samp_trajs = numpy.stack(samp_trajs, axis=0)

    print(samp_trajs)
    input()

    return orig_trajs, samp_trajs, orig_ts, samp_ts

if __name__ == "__main__":
    
    Y_real, Y_fake, X_real, X_fake = generate_spiral2d()

    from matplotlib import pyplot

    ROWS = 12

    fig, axes = pyplot.subplots(nrows=ROWS, ncols=2, sharex=True, sharey=True)

    for row in range(ROWS):
        axes[row,0].plot(Y_real[row,:,0], Y_real[row,:,1])
        axes[row,1].plot(Y_fake[row,:,0], Y_fake[row,:,1], ".")

    axes[0,0].set_title("Ground truth")
    axes[0,1].set_title("Observed data")

    pyplot.show()
