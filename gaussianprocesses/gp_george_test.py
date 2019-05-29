"""
Example from: https://george.readthedocs.io/en/latest/tutorials/model/
Slightly better commented than the example on the website.
"""
from __future__ import division

import numpy as np
import george
from george.modeling import Model
from george import kernels
import emcee
import corner

import sys
import os

import matplotlib.pyplot as plt

np.random.seed(1234)

class Model(Model):
    parameter_names = ("amp", "location", "log_sigma2")

    def get_value(self, t):
        return self.amp * np.exp(-0.5*(t.flatten() - self.location)**2 * np.exp(-self.log_sigma2))

class PolynomialModel(Model):
    parameter_names = ("m", "b", "amp", "location", "log_sigma2")

    def get_value(self, t):
        t = t.flatten()
        return (t * self.m + self.b +
                self.amp * np.exp(-0.5*(t-self.location)**2*np.exp(-self.log_sigma2)))

def lnprob(p, y, model):
    model.set_parameter_vector(p)
    return model.log_likelihood(y, quiet=True) + model.log_prior()

def lnprob2(p, y, gp):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()

def generate_data(params, N, rng=(-5, 5)):

    # Create GP object
    # It needs a kernel to be supplied.
    gp = george.GP(0.1 * kernels.ExpSquaredKernel(3.3))

    # Generate an array for the independent variable.
    # In this case, the array goes from -5 to +5.
    # In case of a spectrum this is the wavelength.
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))

    # Generate the dependent variable array. Flux in case of spectra.
    # The sample method of the gp object draws samples from the distribution
    # used to define the gp object. 
    y = gp.sample(t)  # This just gives a straight line

    # This adds the gaussian "absorption" or "emission" to the straight line
    y += Model(**params).get_value(t)

    # Generate array for errors and add it to the dependent variable.
    # This has a base error of 0.05 and then another random error that
    # has a magnitude between 0 and 0.05.
    yerr = 0.05 + 0.05 * np.random.rand(N)
    y += yerr * np.random.randn(N)  # randn draws samples from the normal distribution

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(t, y, yerr=yerr, fmt='.k', capsize=0)
    plt.show()
    """

    return t, y, yerr

def main():

    truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))
    t, y, yerr = generate_data(truth, 50)

    model = george.GP(mean=PolynomialModel(m=0, b=0, amp=-1, location=0.1, log_sigma2=np.log(0.4)))
    model.compute(t, yerr)

    initial = model.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, model))

    print "Fitting assuming uncorrelated errors."

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    
    print("Running production...")
    sampler.run_mcmc(p0, 1000)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title("Fit assuming uncorrelated errors.")

    # plot data
    ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

    # The positions where the prediction should be computed.
    x = np.linspace(-5, 5, 500)

    # Plot 24 posterior samples.
    samples = sampler.flatchain
    for s in samples[np.random.randint(len(samples), size=24)]:
        model.set_parameter_vector(s)
        ax.plot(x, model.mean.get_value(x), color="#4682b4", alpha=0.3)

    #plt.show()

    # Corner plot for case assuming uncorrelated noise
    tri_cols = ["amp", "location", "log_sigma2"]
    tri_labels = [r"$\alpha$", r"$\ell$", r"$\ln\sigma^2$"]
    tri_truths = [truth[k] for k in tri_cols]
    tri_range = [(-2, -0.01), (-3, -0.5), (-1, 1)]
    names = model.get_parameter_names()
    inds = np.array([names.index("mean:"+k) for k in tri_cols])
    corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels)

    plt.show()  # Seems like this is necessary for the corner plot to actually show up

    # --------------------- Now assuming correlated errors --------------------- #
    print "\n", "Fitting assuming correlated errors modeled with GP noise model."
    kwargs = dict(**truth)
    kwargs["bounds"] = dict(location=(-2, 2))
    mean_model = Model(**kwargs)
    gp = george.GP(np.var(y) * kernels.Matern32Kernel(10.0), mean=mean_model)
    gp.compute(t, yerr)

    # Again run MCMC
    initial = gp.get_parameter_vector()
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(y, gp))
    
    print("Running first burn-in...")
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, 2000)
    
    print("Running second burn-in...")
    p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler.reset()
    p0, _, _ = sampler.run_mcmc(p0, 2000)
    sampler.reset()
    
    print("Running production...")
    sampler.run_mcmc(p0, 2000)

    # Plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.set_title("Fit assuming correlated errors and GP noise model.")

    # plot data
    ax1.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

    # Plot 24 posterior samples.
    samples = sampler.flatchain
    for s in samples[np.random.randint(len(samples), size=24)]:
        gp.set_parameter_vector(s)
        mu = gp.sample_conditional(y, x)
        ax1.plot(x, mu, color="#4682b4", alpha=0.3)

    names = gp.get_parameter_names()
    inds = np.array([names.index("mean:"+k) for k in tri_cols])
    corner.corner(sampler.flatchain[:, inds], truths=tri_truths, labels=tri_labels)

    plt.show()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)