import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import time, warnings, sys, os

global code_version, code_name, article_citation

# Version date - 10/6/2020 - Python 3.8 
code_name = 'icenk'
code_version = '1.01'
article_citation = 'Gerakines, P.A., and Hudson, R. L. 2020. A Modified Algorithm and Open-source Computational Package for the Determination of Infrared Optical Constants Relevant to Astrophysics. ApJ, 901, 52.'

class CalculatedSpectrum:
    ''' Class of objects that contain spectral data (wavenumber x and 
        abosrbance y), including the n & k values used to calculate the 
        spectrum. The fractional deviation from given labSpec at each 
        wavenumber is stored in parameter f. The maximum value of f is stored 
        in parameter m. The last k-correction step taken is stored in parameter 
        dk. '''

    def __init__(self, n, k, param, labSpec, step = None):
        # Initialize the object
        self.x = labSpec.x # array of wavenumbers
        self.y = filmAbsorbance(param, self.x, n, k) # calculated absorbance
        self.n = n # real part of refreactive index
        self.k = k # imaginary part
        self.fringes = filmAbsorbance(param, self.x, param.nvis, 0) # calculated channel fringes
        self.lab = labSpec.y + self.fringes # add fringes to lab spectrum
        self.maxdev = 0. # maximum deviation value
        self.avgdev = 0. # average deviation value
        self.dev = np.ones_like(self.x) # array of deviations
        self.getDeviation() # set values for f, m, avgdev

        self.dk = np.zeros_like(k) # k-step values
        self.dx = np.abs(self.x[1] - self.x[0]) # wavenumber spacing

        # Store resolution of spectrum
        if param.resolution > 0:
            self.res = param.resolution
        else:
            self.res = 2 * self.dx

        self.span = np.max(self.x) - np.min(self.x) # wavenumber range of spectrum
        self.xmin = np.min(self.x) # minimum wavenumber
        self.xmax = np.max(self.x) # maximum wavenumber
        self.xevn = self.x[0::2] # even-indexed wavenumber points
        self.xodd = self.x[1::2] # odd-indexed wavenumber points
        self.fac = 2 * (labSpec.x[1] - labSpec.x[0]) / np.pi # constant factor used in KK integral
        self.checkGoal(param) # check initial values against the goal

        # if k-step value is specified, then use it
        if step is not None:
            self.dk = step

    def update(self, param):
        ''' Updates the calculated spectrum and deviation, checks against goal'''

        self.y = filmAbsorbance(param, self.x, self.n, self.k)
        self.getDeviation()
        self.checkGoal(param)

    def getDeviation(self):
        ''' Calculates the fractional deviation of the current calculated spectrum
            from the lab spectrum and updates the object variables. Performs 
            the calculation in Transmittance, even though the inputs are in
            Absorbance. This avoids division by zero errors.'''

        tlab, tclc = 10.**(-self.lab), 10.**(-self.y)
        self.dev = (tclc - tlab) / (tlab + np.finfo(np.float).eps)
        self.maxdev = np.max(np.abs(self.dev))
        self.avgdev = np.average(np.abs(self.dev))

    def checkGoal(self, param):
        ''' Checks current progress against the goal. Sets flag accordingly.'''

        self.goalreached = self.maxdev <= param.goal
    
    def deltaK(self, param):
        ''' Estimate a change in k that will bring the calculated absorbance 
            spectrum closer to the measured lab absorbance spectrum, using 
            Newton's root-finding method to seek out the root of 
            f(x, n, k) = abs(calculated; based on x, n, k) - abs(lab; given).
            The estimated change delta(k)  is then given by -f(x, n, k)/(df/dk) 
            [where df/dk is the partial derivative of f with respect to k].

            For the derivative of f, this routine uses up to the first 3 terms
            in the analytical expression for ln(Transmission), where the number 
            of terms used depends on the input parameter "approx_terms", which 
            defaults to 1 but may be set by the user in the input file. 

            Returns a 1-D array containing the estimated change in k at each 
            wavenumber.'''

        if param.terms < 0:
            # determine dAbsdk numerically 
            dk = -param.terms 
            a1 = self.y
            a2 = filmAbsorbance(param, self.x, self.n, self.k + dk)
            dAbsdk = (a2 - a1) / dk
        else:
            # determine dlnT/dk using the first 2-3 terms from the analytical 
            # expression for dlnT/dk
            dlnTdk = -4 * np.pi * param.h * self.x
            if param.terms > 1:
                dlnTdk += -2 * self.k / ((1 + self.n)**2 + self.k**2)
                if param.terms > 2:
                    dlnTdk += -2 * (self.k + param.subk) / \
                                   ((param.subn + self.n)**2 + \
                                    (param.subk + self.k)**2) 

            # Convert to Absorbance units
            dAbsdk = -dlnTdk/np.log(10)

        # Abort if the derivative is zero at any wavenumber
        assert not np.any(dAbsdk == 0), \
            "Partial derivative equals zero at 1 or more points. Calculation aborted."

        # Return deltaK = (Mesaured Lab Abs - Calculated Abs) / (dAbs/dk). 
        return (self.lab - self.y) / dAbsdk

    def makeLorentzians(self, param, y, ylimit):
        ''' Returns a lorentzian correction to k, based on current values of n
            and values of parameters lorwid & lorhgt.'''

        x1, x2 = self.xmin + 2*param.lorwid, self.xmax - 2*param.lorwid

        # For each y < ylimit, add a lorentzian to the sum
        lork = np.zeros_like(self.x)
        for i in [j for j in range(len(y)) if y[j] < ylimit 
                  and self.x[j] > x1 and self.x[j] < x2]:

            # Randomize the widths by a few %, while keeping 3*res < wid < 0.2 * (x2-x1)
            wid = putInside(np.random.uniform(low=3*self.res, high=param.lorwid), 3*self.res, 0.2*self.span) 

            # Offset by half of the oscillator's width
            pos = putInside(self.x[i] + 0.5*wid, self.xmin + 2*wid, self.xmax - 2*wid)
            
            # Height depends on y
            hgt = np.abs(ylimit - y[i]) * param.lorhgt

            # Add this Lorentzian to the total
            lork += 0.25 * hgt * wid**2 * \
                    ( 1 / ((self.x - pos)**2 + 0.25 * wid**2) \
                    - 1 / ((self.x + pos)**2 + 0.25 * wid**2) )

        return lork

    def takeKStep(self, param, itr, logs, trial = False):
        ''' Determine and then apply a k correction using deltaK() and  
            optionally lorentzianFix(), calculate new corresponding n values 
            using getDeltaN(), and optionally update the CalculatedSpectrum 
            object with new n and absorbance data using update().  

            Options that affect the process:

                * If the parameter "n_fix" is True and any n < nlimit, an extra 
                  term is added to dk using lorentzianFix().

                * If the keyword "trial" is True, the new n values are 
                  returned without updating the CalculatedSpectrum object. Use 
                  for determining if a step is too large.

                * If the keyword "trial" is False, the CalculatedSpectrum 
                  object is updated with both the new n values and the new 
                  calculated spectrum using the update() method.  

            If trial is False, nothing is returned.
            If trial is True, returns a 1-D array containing the calculated
            set of n values that would correspond to the modified k values.'''

        # Find deltaK using the partial derivative d(lnT)/dk and the current 
        # spectrum, and scale the size of the change according to the value of 
        # the parameter "step". 
        dk = param.step * self.deltaK(param)

        # If the paramter "fix_n" is True, add a Lorentzian to delta-k to 
        # compensate for any current values of n < param.nlimit. 
        if param.fix_n and np.any(self.n < param.nlimit):
             dk += self.makeLorentzians(param, self.n, param.nlimit)

        # Calculate the new set of n values that result from k + dk, using
        # Kramers-Kronig interation of k + dk over all wavenumbers in the 
        # spectrum.
        n = param.nvis + self.getDeltaN(dk) 

        if trial:
            # Return the a 1-D array containing the new n values without 
            # updating other variables.
            return n
        else:
            # Update the CalculatedSpectrum object, and return nothing. 
            self.k += dk
            self.n = n
            self.dk = dk
            self.update(param)
            return

    def getDeltaN(self, dk):
        ''' Calculate the Kramers-Kronig integral of k+dk, using MacLaurin's 
            Formula as described by Ohta & Ishida (1988, Applied Spectroscopy 
            42, 952-957).

            Returns a 1-D array containing the results of the integral for each
            wavenumber in the spectrum.'''
    
        # Integrate odd and even indices separately
        ke, ko = (self.k + dk)[0::2], (self.k + dk)[1::2]
        kkint = np.empty_like(self.k)
        kkint[0::2] = np.array([np.inner((1./(self.xodd - x)) \
                                       + (1./(self.xodd + x)), ko) \
                                for x in self.xevn])
        kkint[1::2] = np.array([np.inner((1./(self.xevn - x)) \
                                       + (1./(self.xevn + x)), ke) \
                                for x in self.xodd])
        kkint[0], kkint[-1] = kkint[1], kkint[-2]

        # Multiply by constant factor and return
        return kkint * self.fac 

class DataLogBundle:
    def __init__(self):
        # Create lists to use as data logs
        self.time = []
        self.fdev = []
        self.step = []
        self.approx = []
        self.avgdev = []

    def appendData(self, f = None, t = None, s = None, a = None, m = None):
        # Append data to desired data logs

        if f is not None:
            self.fdev.append(f)

        if t is not None:
            self.time.append(t)

        if s is not None:
            self.step.append(s)

        if a is not None:
            self.approx.append(a)

        if m is not None:
            self.avgdev.append(m)

class ParameterBundle:
    ''' Class to contain parameters.'''

    def __init__(self, p):
        ''' Parses the strings in dictionary variable p into a new object class.'''

        self.bakf = p['file_bak'] # name of backup file
        self.bakint = int(float(p['bak_interval'])) # number of iterations between back-ups
        self.comment = p['comment'] # comment given in input file
        self.eta = '???' # current ETA for reaching goal
        self.fix_n = p['n_fix'].lower() == 'true' # whether or not to use lorentzian corrections
        self.goal = float(p['goal']) # goal for the maximum deviation 
        self.h = float(p['thickness_cm']) # ice thickness
        self.inputf = p['inputf'] # name of file with input parameters
        self.itmax = int(float(p['iteration_max'])) # maximum number of iterations
        self.laser = float(p['laser_wavelength']) # laser wavelength, in cm
        self.lorhgt = float(p['lorentz_hgt']) # scaling factor for lorentzians
        self.lorwid = float(p['lorentz_wid']) # scaling factor for widths of lorentzians
        self.nlimit = float(p['n_limit']) # lowest allowed value of n before lorentzian correction is applied
        self.nvis = float(p['visible_index']) # visible refractive index of ice
        self.outputf = p['file_output'] # output file name
        self.plotIter = 0 # the number of plots produced so far
        self.plotLastIter = -1 # the last iteration that was plotted
        self.plotInterval = int(float(p['plot_interval'])) # plotting interval, in iteration steps
        self.figureNumber = 1 # used to control matplotlib figure
        self.plotsize = float(p['plot_size']) # width, height of plot window
        self.resolution = float(p['resolution']) # resolution of spectrum
        self.spectrumf = p['file_spectrum'] # name of file containing the lab spectrum 
        self.startf = p['file_start'] # name of a file containing n, k starting values
        self.step = float(p['step']) # size of k-step
        self.step_adapt = p['step_adapt'].strip().lower() == 'true' # use adpative step sizes?
        self.step_dnrate = float(p['step_dnrate']) # rate at which step size is reduced
        self.step_interval = int(float(p['step_interval'])) # iteration interval between changes to step size
        self.step_uprate = float(p['step_uprate']) # rate at which step size is increased 
        self.stepmax = float(p['step_max']) # maximum step size
        self.stepmin = float(p['step_min']) # minimum step size
        self.substratef = p['file_substrate'] # name of file containing substrate n and k
        self.terms = float(p['approx_terms']) # number of terms to use in partial derivative dlnT/dk
        self.xr1 = p['xrange1'] # wavenumber on left side of x axis
        self.xr2 = p['xrange2'] # wavenumber on right side of x axis

        # Amount of detail to be shown in the plot
        a = str(p['plot_detail']).lower()
        if a == 'none' or a == '0':
            self.plot_detail = 0
        elif a == 'high' or a == '2':
            self.plot_detail = 2
        else:
            self.plot_detail = 1

class Spectrum:
    ''' Object class to hold the laboratory-measured spectrum.'''

    def __init__(self, x0, x, y, f):
        self.x0 = x0 # Original input wavenumber array
        self.x = x # Evenly spaced wavenumber array
        self.y = y # Absorbance array interpolated onto evenly spaced x array
        self.xmax = np.max(x) # highest wavenumber in spectrum
        self.xmin = np.min(x) # lowest wavenumber in spectrum 

def addDeltaPlot(ax, x, d, pc, pa, lb):
    ''' Adds d(x) to plot axes ax that already contains data, and 
        scales d(x) to be easily visible. Also adds label (lb) to the 
        plot legend that includes the scaling factor. Passed parameter pc 
        contains the line's color, and pa contains the line's alpha value.'''

    # Determine the scaling factor. 
    scale_fac = 1
    y1, y2 = ax.get_ylim()
    d1, d2 = np.min(d), np.max(d)
    if d1 != d2:
        scale_fac = 10**np.ceil(np.log10(0.1*(y2-y1)/(d2-d1)))

    if not np.isfinite(scale_fac):
        return ax.get_ylim()
        
    # Create a legend label that includes the scaling factor.
    sf = '%d' % scale_fac if scale_fac <= 1000 else r'10$^{%d}$' % int(np.log10(scale_fac))
    l = '' if scale_fac == 1 else r' $\times$ %s' % sf

    # Add the line to the plot.
    ax.plot(x, scale_fac * d, linestyle = 'solid', color = pc, alpha = pa, 
            label = lb + l, zorder = 1)

    # Change the limits of the y axis, if necessary, and return the new limits.
    y1, y2 = min([scale_fac * d1, y1]), max([scale_fac * d2, y2])
    return ax.set_ylim(y1 - 0.05*(y2-y1), y2 + 0.05*(y2-y1))

def filmAbsorbance(param, x, n, k):
    ''' Combines current values of n and k with other known parameters and 
        calculated the absorbance (-log10 T) for a layer of thickness h (in cm)
        and index m1, between infinite media with indices m0 and m2.  Assumes 
        normal incidence (angle = 0 degrees).  Transmission of the bare 
        substrate is removed. Channel fringes are NOT removed.  [See, e.g., 
        O.S. Heavens (1955), Optical Properties of Thin Solid Films, Chapter 4, 
        pp. 51-57.]

        Returns a 1-D array containing the absorbance at each wavenumber in 
        x.'''

    m0, m1, m2 = 1, n - 1j*k, param.subn - 1j*param.subk # complex refractive indices

    # Fresnel coefficients
    t1 = 2 * m0 / (m0 + m1) # transmission coefficient at 0-1 interface
    t2 = 2 * m1 / (m1 + m2) # transmission coefficient at 1-2 interface
    t3 = 2 * m0 / (m0 + m2) # transmission coefficient of bare substrate
    r1 = (m0 - m1) / (m0 + m1) # reflection coefficient at 0-1 interface
    r2 = (m1 - m2) / (m1 + m2) # reflection coefficient at 1-2 interface

    d = 2*np.pi*x*param.h*m1 # complex phase change - assumes incidence angle is 0 degrees (normal)
    tr = (t1 * t2 / t3) * np.exp(-1j * d) / (1 + r1 * r2 * np.exp(-2j * d)) # (ice + substrate) / substrate
    transm = (tr * np.conj(tr)).real 

    return -np.log10(transm)

def findNK(infile_name):
    ''' From a given laboratory absorbance spectrum measured in transmission 
        through a thin film, that ice's refractive index at a visible 
        wavelength, and the film's thickness, extract a set of optical 
        constamts (n, k) that reproduce the spectrum. The file infile_name 
        contains all of the necessary information and optional parameters
        for executing the calculation. See the file input-definitions.csv for a
        description of each parameter and its use. '''

    time0 = time.time() # store the current clock time, in seconds
    
    printl('\n## START ## ' + code_name + ' v' + code_version + ' -- ' + time.asctime() + ' ##')

    # Read data from files
    param = readParameters(infile_name)
    if not param:
        printl("\nStopping.")
        return

    labSpec = readSpectrum(param.spectrumf)

    if not labSpec:
        printl("\nStopping.")
        return

    if not readSubstrate(labSpec.x, param):
        printl('\nStopping.')
        return

    printl('\n+ Starting calculations...')

    logs = DataLogBundle()
    
    # Make sure step parameter agrees with stepmin and stepmax
    param.step = putInside(param.step, param.stepmin, param.stepmax)

    # Load and plot the initial state
    time1, itr, cSpec = setInitialState(labSpec, param, logs)

    if param.fix_n and param.lorwid == 0:
        param.lorwid = 20*cSpec.res

    # Keep track of the number of resets
    reset_count = 0 

    # Iterate until the maximum deviation is smaller than the goal, or the  
    # number of iterations surpasses the allowed maximum.
    while not cSpec.goalreached and itr < param.itmax:
        itr += 1

        # Modify the k correction step size, if requested by step_adapt.
        param.step += modifyStep(param, logs, itr)

        # Make the k correction; find new n-values and new calculated spectrum.
        cSpec.takeKStep(param, itr, logs)
  
        # Check for infinity, NaN
        if not np.all(np.isfinite(cSpec.k)) and not np.all(np.isfinite(cSpec.n)):
            printl('\n    !! Infinity or NaN detected in values of n or k')
            printl('    !! CALCULATION ABORTED')
            break

        # Append current data to the logs
        logs.appendData(f = cSpec.maxdev, t = time.time()-time1, s = param.step, m = cSpec.avgdev)

        # Plot the results of this iteration step.
        plotResults(itr, param, cSpec, logs)

        # Make a backup of the current results.
        if itr > 0 and itr % param.bakint == 0:
            writeData(labSpec, cSpec, param, bak = True)

        # Print a message after the first 10 iteration steps.
        if itr == 10:
            printl('    Each iteration is taking about %.2g s.' \
                  % (logs.time[-1]/itr))
    
    # Iterations are finished - report the final results.
    time2 = time.time()

    plotResults(itr, param, cSpec, logs, final = True)

    s = '\n+ Results:'

    # Report success or failure.
    s2 = ' GOAL REACHED.' if cSpec.goalreached else ' FAILED TO REACH GOAL.'

    printl(s + s2)

    if itr >= param.itmax: 
        printl('    EXCEEDED ALLOWED NUMBER OF ITERATIONS.')

    printl('    Max frac. dev. = %.2e after %d iterations'  % (cSpec.maxdev, itr))
    printl('    Avg frac. dev. = %.2e'  % cSpec.avgdev)

    # Report runtime statistics. 
    s = '    Elapsed time = %s' % timeString(time.time() - time0)
    if itr>0:
        s += ' (%.3f s/iter)' % ((time2 - time1) / (itr + np.finfo(np.float).eps))
    printl(s)

    # Write the n & k data and header information to file outputf.
    printl('    Storing [wavenumber, absorbance, n, k] results in file "' + \
        param.outputf + '".')
    writeData(labSpec, cSpec, param)

    

def getLogLimits(y, low = 1.0e-38, high = 1.0e+38, padding = None):
    ''' Calculates appropriate limits for plotting y on a logarithmic axis, with
        minimum and maximum allowed values specified. Avoids error messages 
        associated with non-positive or infinite values of y.

        Returns a tuple containing the limits (min, max)'''

    q = [v for v in y if (v > low and v < high)]
    if len(q) > 0:
        y1, y2 = 10.0**np.floor(np.log10(min(q))), 10.0**np.ceil(np.log10(max(q)))
        if padding is not None:
            y1 /= 10.**(padding[0]*np.log10(y2/y1)) 
            y2 *= 10.**(padding[1]*np.log10(y2/y1)) 

        return y1, y2
    else:
        return 0.1, 1

def getETA(t, y, y0, endpts = False):
    ''' Calculate the estimated time remaining until the goal of y0 is reached, 
        using the data points given in arrays t & y. 

        Returns a formatted string containing the estimated time. '''

    # Use a simple linear least-squares fit to y(t)
    n = len(t)
    sx, sx2, sy, sxy = np.sum(t), np.sum(t*t), np.sum(y), np.sum(t*y)
    slope = (n*sxy - sx*sy)/(n*sx2 - sx**2)

    if slope == 0:
        return '???'

    intercept = (sx2*sy - sx*sxy)/(n*sx2 - sx**2)
    t0 = (y0 - intercept)/slope
    eta = t0 - t[-1]

    if eta < 0:
        seta = '???'
    else:
        seta = timeString(eta)

    if endpts:
        pts = slope*np.array([t[-1], t0]) + intercept
        return seta, np.array([t[-1], t0]), np.array([pts[0], pts[1]])

    return seta

def modifyStep(param, logs, itr):
    ''' Calculates a change in the step size based on current 
        progress over the last set of iterations (# of iterations in 1 set = 
        step_interval) and compares it to the trend over the past 20 sets of 
        iterations (20 * step_interval).

        No result is returned if the input parameter step_adapt is set to False.
 
        The size of the returned result is determined by the input parameters 
        step_uprate and step_dnrate, and modified to ensure that the total step
        size stays within the range [stepmin, stepmax]. 

        Returns the calculated change in step size.'''
 
    if (not param.step_adapt) \
        or (itr % param.step_interval) > 0 \
        or itr < param.step_interval*2:
        return 0

    # Determine the progress over the last # of iterations = 20 * step_interval
    i0 = max([0, itr - 21*param.step_interval])
    i1 = max([0, itr - param.step_interval])
    favg = np.average(np.log10(logs.fdev[i0:i1])) # average
    fdev = np.std(np.log10(logs.fdev[i0:i1])) # std deviation

    # Compare to the progress over the last # of iterations = 1 * step_interval
    fcur = np.average(np.log10(logs.fdev[i1:]))
    dfd = favg - fcur # amount by which dev has decreased

    dfd /= fdev if fdev > 0 else (np.abs(dfd) + np.finfo(np.float).eps) 

    if np.abs(dfd) > 0.5: 
        # Change the step size
        dstep = param.step_uprate if dfd > 0 else -param.step_dnrate
    else:
        # There has been very little change, in either direction. 
        # Try a small change in the step size in a randomly chosen direction.
        dstep = 0.1 * np.random.choice([param.step_uprate, -param.step_dnrate])

    return putInside(param.step * (1 + dstep), param.stepmin, param.stepmax) - param.step

def parameterDefaults(fname):
    # Create a dictionary containing default paramters
    return dict(
        # how many terms to use in approximation of dlnT/dk, default = 1
        approx_terms = 1,

        # number of iteration steps between writing a backup of current 
        # results to file_bak
        bak_interval = 10,
        file_bak = code_name + '-bak.tmp',

        # A string to be written to the output file
        comment = '', 

        # the name of the output file 
        file_output = code_name + '.out',

        # the name of the spectrum from which n & k are to be derived
        file_spectrum = 'spectrum.dat',

        # the name of an ASCII file w/columns [x], [Abs], [n], [k] 
        # containing the initial values of n & k
        file_start = '',  

        # the name of an ASCII file w/columns [x], [n], [k] containing the 
        # values of n(x) and k(x) for the substrate that was used to 
        # measure the input spectrum
        file_substrate = 'substrate.dat', 

        # the calculation stops when the max deviation falls below this 
        # value
        goal = 1.e-3,

        # the maximum number of iterations allowed
        iteration_max = 10000, 

        # the name of the input file
        inputf = fname,

        # the wavelength (in cm) of the laser used to measure the
        # thickness of the sample - used with thickness_fringes to 
        # determine the sample's thickness in cm
        laser_wavelength = 0.67e-4,

        # parameters used to make a correction to k when negative
        # n-values are encountered (used with n_fix and n_limit)
        lorentz_hgt = 0.01, # default lorentian height, as a fraction of |nlimit-n|
        lorentz_wid = 0, # width in units of wavenumbers, "0" for default of 20 resolution elements

        # if True, attempt to compensate for values of n below n_limit
        n_fix = 'False',

        # minimum value of n(x) allowed before a correction is applied
        n_limit = 0, 

        # level of detail to plot. 
        # Allowed values are 'none' (or 0), 'default' (or 1), 'high' (or 2)
        plot_detail = 1,

        # number of iteration steps between updates to the data plots
        plot_interval = 1,

        # Plot window height, in inches. Default is 10
        plot_size = 10.,

        # resolution (in cm-1) of the input absorbance spectrum. 
        # if 0 (default), then 2*x-spacing is assumed.
        resolution = 0,

        # (initial) fraction of the k-correction to be applied at each step
        step = 0.95,

        # if True, attempt to modify step according to current performance,
        # used with step_uprate, step_dnrate, and step_interval
        step_adapt = 'False',

        # determines how quickly the step size is decreased or increased 
        # used with step_adapt
        step_dnrate = 0.02,
        step_uprate = 0.01,

        # number of iteration steps between attempts to modify the step
        step_interval = 2, 

        # the maximum and minimum values allowed for the step parameter
        step_max = 0.95, 
        step_min = 0.001,

        # thickness of the laboratory-measured ice sample, in number of 
        # laser fringes or in centimeters. Used with laser_wavelength.
        thickness_fringes = -1, 
        thickness_cm = 1e-4,

        # known refractive index of the ice sample at the laser wavelength
        visible_index = 1.33,

        # wavenumber range [high, low] over which to plot the spectra
        xrange1 = 'default', 
        xrange2 = 'default'
        )

def plotResults(itr, param, cSpec, logs, final = False):
    ''' Opens the graphics window or updates it using the current data and based
        on the input parameter plot_detail. 

        Possible plots (from top to bottom):
            Panel 1: n, k vs. wavenumber
            Panel 2: lab spectrum, calculated spectrum vs. wavenumber
            Panel 3: fractional deviation (log scale) vs. wavenumber
            Panel 4: max dev, step (log scale) vs. iteration number. 

        Effects of the input parameter "plot_detail":
            plot_detail = 0 ("none"): No plot window
            ## plot_detail = 1 ("low"): Panels 1 & 2 only
            plot_detail = 1 ("default"): Panels 1-3
            plot_detail = 2 ("high"): Panels 1-4
                * add plot of dk vs. wavenumber to Panel 1
                * add plot of difference vs. wavenumber to Panel 2, 
                * add information about lorentz corrections in k to Panel 4.

            making none = 0
            making default = 1 (from 3)
            making high = 2 (from 4)


        '''

    # Return immediately if plot_detail is 0, or if either n or k is not finite
    if param.plot_detail == 0 or (itr % param.plotInterval and not final) \
    or not np.all(np.isfinite(cSpec.n)) or not np.all(np.isfinite(cSpec.k)):
        return

    # Set the plot colors 
    pl.style.use('dark_background')
    plotalpha = 0.75
    plotcolors = {'blue':'dodgerblue', 
                  'darkgreen':'darkgreen', 
                  'gray':'0.25', 
                  'green':'green', 
                  'lightgray':'0.75', 
                  'magenta':'magenta', 
                  'red':'red', 
                  'white':'white', 
                  'yellow':'yellow'}

    # Set the default properties for plot components

    kw_tick_params = {'axis': 'both', 
                      'labelsize': 'small', 
                      'which': 'both'}

    kw_grid = {'color': plotcolors['gray'], 
               'which': 'both', 
               'zorder': 0}

    kw_label = {'fontsize': 'medium'}

    kw_legend = {'fontsize': 'small'}

    kw_title = {'fontsize': 'medium'}

    kw_line1 = {'color': plotcolors['white'], 
                'linestyle': 'solid', 
                'zorder': 2}

    kw_line2 = {'color': plotcolors['red'], 
                'linestyle': 'solid', 
                'zorder': 3}

    kw_line3 = {'color': plotcolors['blue'], 
                'linestyle': 'solid', 
                'zorder': 3}

    kw_line4 = {'color': plotcolors['green'], 
                'linestyle': 'solid', 
                'zorder': 3}

    kw_line5 = {'color': plotcolors['magenta'], 
                'linestyle': 'solid', 
                'zorder': 3}

    kw_nlimit = {'color': plotcolors['yellow'], 
                 'linestyle': '-', 
                 'alpha': 0.5, 
                 'zorder': 1}

    kw_nvis = {'label': r'n$_{\rm vis}$', 
               'color': plotcolors['lightgray'], 
               'linestyle': '-', 
               'alpha': 0.5, 
               'zorder': 1}

    # On the first function call, set the initial and/or unchanging parameters.
    # Later function calls retrieve the stored information about the figure.
    if param.plotIter == 0:
        # Determine the wavenumber range for plotting spectra
        # This range may be set with input parameters xrange1 and xrange2
        if param.xr1 == 'default':
            param.xr1 = cSpec.xmax # left x-limit
        else:
            param.xr1 = float(param.xr1)

        if param.xr2 == 'default':
            param.xr2 = cSpec.xmin # right x-limit
        else:
            param.xr2 = float(param.xr2)

        # Make sure there are points within the chosen plot range
        # If not, then default to the entire range of x data
        if len(np.where((cSpec.x <= param.xr1) * \
                        (cSpec.x >= param.xr2))[0]) == 0:
            param.xr1, param.xr2 = (cSpec.xmax, cSpec.xmin)

        # Determine if a figure window already exists. If so, do not move or resize it
        fig_exists = pl.fignum_exists(1)

        # Create/take control of the figure window
        fig = pl.gcf() 

        # Store the figure number for later use
        param.figureNumber = fig.number

        # Set the window title
        s = os.path.basename(param.spectrumf) + ' - '
        s +=  code_name + ' v' + code_version
        fig.canvas.set_window_title(s)
                
        # Turn on interactive plotting
        pl.ion()

        # Define the properties of the panel subplots - at least 3 panels needed
        faspect = 5.25 / 3.5
        gs1 = mpl.gridspec.GridSpec(4, 1, 
                                    hspace = 0, 
                                    right = 0.95, 
                                    top = 0.95, 
                                    bottom = 0.08)
        panel0 = pl.subplot(gs1[0]) 
        panel1 = pl.subplot(gs1[1])
        panel2 = pl.subplot(gs1[2])
        panel3 = pl.subplot(gs1[3])

        # Set the tick label properties
        panel0.set_xticklabels([]) # no x ticklabels
        panel1.set_xticklabels([])
        panel2.set_xticklabels([])

        panel0.tick_params(**kw_tick_params) 
        panel1.tick_params(**kw_tick_params) 
        panel2.tick_params(**kw_tick_params)
        panel3.tick_params(**kw_tick_params)

        # Turn on the grid lines
        panel0.grid(**kw_grid)
        panel1.grid(**kw_grid)
        panel2.grid(**kw_grid)
        panel3.grid(**kw_grid)

        panel0.set_axisbelow(True)
        panel1.set_axisbelow(True)
        panel2.set_axisbelow(True)
        panel3.set_axisbelow(True)

        # Add y-axis labels
        panel0.set_ylabel('n', **kw_label)
        panel1.set_ylabel('k', **kw_label)
        panel2.set_ylabel('Absorbance', **kw_label)
        panel3.set_ylabel('Deviation', **kw_label)

        # Use a logarithmic y scale in panel3
        panel3.set_yscale('log')

        # Set the x-axis limits
        panel0.set_xlim(param.xr1, param.xr2)
        panel1.set_xlim(param.xr1, param.xr2)
        panel2.set_xlim(param.xr1, param.xr2)
        panel3.set_xlim(param.xr1, param.xr2)

        # Set x-axis label in panel3
        panel3.set_xlabel('Wavenumber (cm' + r'$^{-1}$' + ')', **kw_label)

        if param.plot_detail > 1:
            # All 5 panels needed.
            faspect = 6.25 / 3.5

            # Set the top 4 panels to use 70 % of the vertical space
            gs1.update(bottom = 0.3)

            # Add panel4
            gs2 = mpl.gridspec.GridSpec(1, 1, 
                                        right = 0.95, 
                                        bottom = 0.05, 
                                        top = 0.25)

            panel4 = pl.subplot(gs2[0])
            panel4.set_xlabel('Iteration #', **kw_label)
            panel4.set_yscale('log')
            panel4.tick_params(**kw_tick_params)     
            panel4.yaxis.set_label_position('left')
            panel4.set_ylabel('Max Deviation, Step Size', **kw_label)
            panel4.grid(**kw_grid)
            panel4.set_axisbelow(True)
        

        # Set the height and width, in inches
        fig.set_size_inches(param.plotsize/faspect, param.plotsize)

        # Add a title to the top of the figure
        fig.suptitle('Initial data: dev= %.2e' % cSpec.maxdev, **kw_title)

    else: # This is not the first time plotting
        # Reload the figure
        fig = pl.figure(param.figureNumber)

        # Get the list of axes from the figure & remove old data 
        axs = fig.get_axes()
        for a in axs:
            for artist in a.lines + a.texts + a.collections:
                artist.remove()
        
        # Format the figure title. First add time elapsed, number of 
        # iterations, max deviation
        s = '%s' % timeString(logs.time[-1])

        if not final:
            # Add an estimate of the remaining time to the title
            i0 = 0 if itr < 20 else int(np.floor(0.5 * itr))
            param.eta = getETA(np.array(logs.time[i0:]),
                               np.log10(np.array(logs.fdev[i0:])), 
                               np.log10(param.goal))

            s +=  ' (%s)' % param.eta 

        s += ': it= %d, dev= %.2e' % (itr, cSpec.maxdev)

        if param.step_adapt:
            # Add current step size to the title
            s += ', step= %.3g' % (param.step)

        if param.fix_n:
            # Add minimum n value to the title
            s += ', min(n)= %.3g' % (min(cSpec.n))
            
        # Add the title to the plot
        fig.suptitle(s, **kw_title)

        # Assign axis variables according to value of plot_detail
        panel0, panel1, panel2, panel3 = axs[0], axs[1], axs[2], axs[3]
        if param.plot_detail > 1: 
            panel4 = axs[4]
    
    # Add data to the panels.
    pii, = np.where((cSpec.x >= param.xr2)*(cSpec.x <= param.xr1))
    px = cSpec.x[pii]

    # Panel 0: n vs. wavenumber
    panel0.plot(px, cSpec.n[pii], label = 'n', **kw_line2)
    setAxisYLimits(panel0, cSpec.n[pii]) # set y limits
    panel0.axhline(param.nvis, **kw_nvis) # include a line for nvis
    panel0.legend(loc='upper left', **kw_legend) # create a legend

    if param.plot_detail > 1 and param.fix_n:
        # Add a horizontal line at y = param.nlimit in panel0 
        panel0.axhline(y=param.nlimit, **kw_nlimit)

    # Panel 1: k vs. wavenumber
    panel1.plot(px, cSpec.k[pii], label = 'k', **kw_line3)
    setAxisYLimits(panel1, cSpec.k[pii])

    if param.plot_detail > 1:
        # Include delta(k) in panel1
        addDeltaPlot(panel1, px, cSpec.dk[pii], 
                     plotcolors['lightgray'], plotalpha, r'$\Delta$k')

    panel1.legend(loc='upper left', **kw_legend) # create a legend

    # Panel 2: lab spectrum and calculated spectrum vs. wavenumber
    _lab = cSpec.lab[pii] - cSpec.fringes[pii]
    _clc = cSpec.y[pii] - cSpec.fringes[pii]
    panel2.plot(px, _clc, label = 'calc', **kw_line5)
    panel2.plot(px, _lab, label = 'input', **kw_line4)
    setAxisYLimits(panel2, np.append(_lab, _clc)) 

    if param.plot_detail > 1:
        # Plot the difference between calculated and lab spectra in panel 2
        addDeltaPlot(panel2, px, _clc-_lab, 
                     plotcolors['lightgray'], plotalpha, r'$\Delta$')

    del _lab, _clc

    # Create the legend for panel 2
    panel2.legend(loc='upper left', **kw_legend)

    # Panel 3: fractional deviation vs. wavenumber 
    # Find appropriate log limits for y-values inside the plot x-range
    _dv = np.abs(cSpec.dev[pii])
    panel3.plot(px, _dv, label = 'deviation', **kw_line1)
    panel3.set_ylim(param.goal/10, getLogLimits(_dv)[1])
    panel3.axhline(param.goal, label = 'goal', **kw_line4)

    if param.plot_detail > 1:
        # Add the point of maximum fractional deviation
        mxi = np.argmax(np.abs(cSpec.dev))
        panel3.plot(cSpec.x[mxi], np.abs(cSpec.dev)[mxi], 'r+', label='max') 

    # Add a legend 
    panel3.legend(loc='upper left', **kw_legend)

    if param.plot_detail > 1: # Add panel 4
        panel4.set_xlim(0, itr*1.1) # x-axis is iteration number
        p4x = list(range(len(logs.fdev)))

        # find limits by sending log lists to getLogLimits
        panel4.set_ylim(getLogLimits(logs.fdev + logs.step + logs.avgdev,
                                     padding = (0.1, 0)))
            
        # Plot the maximum deviation vs. iteration number
        panel4.plot(p4x, logs.fdev, label = 'maxd %.3g' % logs.fdev[-1], 
                    **kw_line1)

        # Plot the average deviation vs. iteration number
        panel4.plot(p4x, logs.avgdev, label = 'avgd %.3g' % logs.avgdev[-1], 
                    **kw_line5)

        # Plot the step size vs. iteration number
        panel4.plot(p4x, logs.step, label = 'step %.3g' % logs.step[-1], 
                    **kw_line3)

        del p4x

        panel4.axhline(param.goal, color='g', label = 'goal %.3g' % param.goal)
        
        # Create the plot legend
        panel4.legend(loc='lower left', **kw_legend)

    if final: # If this is the final plot, change figure title
        s = os.path.basename(param.spectrumf) + ' - '
        s += code_name + ' v' + code_version
        fig.canvas.set_window_title(s + '- Results') 
        fig.suptitle('Result: dev= %.2e after %d iteration steps in %s' 
                     % (cSpec.maxdev, itr, timeString(logs.time[-1])), 
                     **kw_title)

    # Flush the drawing events
    fig.canvas.flush_events()

    # Use the appropriate command to display/redraw the figure
    if itr == 0:
        pl.show(block=False)

    # A slight delay is needed so that the window remains responsive.
    # This snippet of code creates a short pause without changing the window 
    # focus (which the "pause" function does).
    backend = pl.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(0.001)

    param.plotLastIter = itr
    param.plotIter += 1
    return

def printl(s):
    print(s)
    with open('icenk.log', 'a') as f:
        f.write(s + '\n')

def putInside(a, b, c):
    #  if min(b, c) < a < max(b, c), return a
    #  if a <= min(b, c), return min(b, c)
    #  if a >= max(b, c), return max(b, c)
    
    return min(max(a, min(b, c)), max(b, c))

def readFile(f):
    ''' Uses the numpy.loadtxt routine to read data from an ASCII file. 
        Assumes plain text. By default, this routine assumes the delimiter is 
        whitespace (spaces or tabs), unless the filename ends with CSV, in which
        case a comma is assumed as the delimiter. '''
        
    delim = None
    if f.lower().endswith('csv'):
        delim = ','

    try:
        a = np.loadtxt(f, unpack=True, delimiter=delim)
    except:
        printl("    ERROR READING FILE.")
        return []

    return a

def readParameters(fname):
    ''' Reads parameters from the input file given by fname.  The file must be 
        in plain-text format. If the file extension of fname is .csv, then it 
        is assumed that the entries on each line are delimited by commas. For 
        every other file extension, whitespace delimiters (spaces or tabs) are 
        assumed. If a line begins with "##", that line is ignored 
        completely. Every other line of the file must have 2 entires, e.g., 
        "<keyword> <value>". If a line does not begin with a known keyword, 
        that line is printed but otherwise ignored. The case of the keyword is 
        ignored (e.g., "COMMENT" is equivalent to "comment"). 

        A dictionary is created and used to create a ParameterBundle variable, 
        which is returned.
        '''
    
    printl('\n+ Reading input parameters from "%s"...' % fname)

    try:
        f = open(fname, "r")
    except:
        printl('    ERROR READING INPUT FILE')
        return False

    # Assume that entries on each line of the input file are separated by
    # whitespace, unless the file extension is .csv
    sep = None
    if fname.lower().endswith('.csv'):
        sep = ','

    # Create dictionary of default parameters
    p = parameterDefaults(fname)

    # Read each line of the input file and replace dictionary entries when the 
    # keyword is present as the first entry. Skip lines beginning with '##' or 
    # unknown keywords.
    for line in [l for l in f if len(l) > 1]:
        s = line.split(sep) # split line into strings for keyword & value
        r = s[0].strip().lower() # convert keyword string to all lowercase

        if len(s) > 1:
            q = s[1].strip()

            # If the first entry is "comment", then join all other entries on
            # this line into one. This corrects for the treatment of commas in 
            # csv files
            if r == 'comment':
                _s = sep
                if sep is None:
                    _s = ' '

                q = _s.join(['%s' % v for v in s[1:]])
        
        if r[0:2] == '##':
            # Skip this line
            pass
        elif r == '#':
            # Prints out to the screen during program execution
            printl('     ' + line[2:])
        else:
            # Replace dictionary values for known keywords. Skip unknown 
            # keywords and print a message to the screen.
            if r in p:
                p[r] = q
                printl('  > %s: %s' % (r, q))
            else:
                printl('  WARNING - Unrecognized keyword "%s". Skipping.' % r)

    f.close()
    del s, r, q

    # If the number of fringes is specified, overwrite the existing 
    # thickness with the value derived from the number of fringes
    if float(p['thickness_fringes']) > 0: 
        p['thickness_cm'] = 0.5 * float(p['thickness_fringes']) \
                                 * float(p['laser_wavelength']) \
                                 / float(p['visible_index'])

    return ParameterBundle(p)

def readSpectrum(fname):
    ''' Reads the (x, y) data from the file given by fname, converts it to an 
        even grid in the range [min(x), max(x)], then converts the y data to 
        absorbance if it is found to be in transmittance.   

        Returns a Spectrum object.'''

    printl('\n+ Reading spectrum from "%s"...' % fname)

    fd = readFile(fname)
    if not np.size(fd): # file either doesn't exist or contains no data
        return False

    xx, yy = fd
    del fd

    # Sort arrays in order of increasing values of xx
    ind = np.argsort(xx) 
    xx, yy = xx[ind], yy[ind]

    printl("    %d points in spectrum." % len(xx))
    printl("    Wavenumber range = [%.2f, %.2f]" % (np.max(xx), np.min(xx)))

    # Convert to an even grid with constant x spacing
    dx = min(np.diff(xx))
    x = np.linspace(np.min(xx), np.max(xx), int(xx.ptp()/dx)+1, dtype=float)
    y = np.interp(x, xx, yy)
    
    del ind, yy
 
    if np.abs(np.diff(xx).ptp()) > 0:
        printl("    Wavenumber spacing is not constant. (%.3f to %.3f)" % (np.min(np.diff(xx)), np.max(np.diff(xx))))
        printl("    Converting to a regular wavenumber grid.")
        printl("    New wavenumber spacing = %.3f" % (x[1]-x[0]))
        printl("    %d points in the new grid." % len(x))
    else:
        printl("    Wavenumber spacing = %.3f" % (x[1]-x[0]))

    # Determine if the y data are in absorbance or transmittance units
    if y[0] > 0.1 and y[-1] > 0.1: 
        # If both endpoints have values > 0.1 then the units are probably transmittance (I/I_0)
        printl("    The y-data seem to be in Transmittance units.")
        printl("    Converting the y-data to Absorabnce, -log_10(T).")
        y = -np.log10(y) # convert to absorbance units
        if y[0] > 1.0:
            # at least one endpoint is still quite large, the file may have been in % transmission, 
            # so remove an additional factor of 100 from the data
            y -= 2.0
    else:
        # Endpoints have small values, consistent with baselines in absorbance 
        printl("    The y-data seem to be in Absorbance units.")

    # Create a Spectrum object with these x, y data and return it
    return Spectrum(xx, x, y, fname)

def readStartfile(x, f):
    ''' Reads in a previous n, k result from file f and use it as the 
        starting point for this calculation. Interpolates the data in f onto 
        the wavenumber grid given in x.'''

    printl("    Loading previous results from %s" % f)

    # Attempt to read the file
    fd = readFile(f)

    if not np.size(fd):
        # File doesn't exist or contains no data
        return None, None

    t, s, n, k = fd
    del fd

    # Interpolate n, k data onto x and return the results
    return np.interp(x,t,n), np.interp(x,t,k)

def readSubstrate(w, param):
    ''' Reads the substrate's wavenumber, n, k data from file param.substratef 
        and interpolates the data onto the wavenumber grid given in the array w.
        Adds the substrate data to the param object. Returns False if the
        file doesn't exist. Returns True if data read successfully.'''

    printl('\n+ Reading substrate data from "%s"...' % param.substratef)

    # Attempt to read from the file
    fd = readFile(param.substratef)

    if not np.size(fd):
        # File doesn't exist or contains no data
        return False

    x, y, z = fd
    del fd

    # Sort by x
    ind = np.argsort(x)
    x, y, z = x[ind], y[ind], z[ind]

    printl("    %d points in file." % len(x))
    printl("    Wavenumber range = [%.2f, %.2f]" % (np.max(x), np.min(x)))

    # Interpolate onto w and store in param object
    param.subn = np.interp(w, x, y)
    param.subk = np.interp(w, x, z)

    return True

def setAxisYLimits(ax, dat, margin=0.05):
    d1, d2 = np.min(dat), np.max(dat)
    d = margin*(d2-d1)
    return ax.set_ylim(d1-d, d2+d)

def setInitialState(labSpec, param, logs):
    ''' Sets n, k to initial values and finds the calculated spectrum 
        based on those values. Returns the initial time, iteration number, and 
        CalculatedSpectrum object.'''

    # If indicated, load the initial values of n(x) & k(x) from a file.
    # Otherwise, use the default initial values n(x) = nvis, k(x) = 0.
    if param.startf != '':
        n, k = readStartfile(labSpec.x, param.startf)
    else:
        n, k = param.nvis * np.ones_like(labSpec.x), np.zeros_like(labSpec.x)
    
    # Calculate the initial spectrum based on this set of n & k.
    cSpec = CalculatedSpectrum(n, k, param, labSpec, step = k)
    
    if cSpec.maxdev > param.goal:
        # Check the results of a k-correction. If it produces values of n below
        # nlimit, then reduce the initial value of step. This prevents overstepping
        # on the first iteration, but takes some extra time (depending on the 
        # number of points in the spectrum).
        _f, _step = False, param.step
        if param.step_adapt and param.fix_n:
            _n = cSpec.takeKStep(param, 0, None, trial = True)
            while np.min(_n) < param.nlimit and \
                  param.step > param.stepmin / 0.5:
                
                # Reduce the step size and try again
                param.step *= 0.5 

                if not _f:
                    printl('    Initial n-values < nlimit.')
                    
                _f, _n = True, cSpec.takeKStep(param, 0, None, trial = True)

            if _f:
                printl("    Initial step size decreased from %.2g to %.2g" % (_step, param.step))
    else:
        # No further calculations are needed
        printl("    Initial results already meet the specified goal.")

    # Set the initial values for the time and iteration number.
    time1, itr = time.time(), 0

    # Start the logs (used for plotting).
    logs.appendData(t = time.time() - time1, f = cSpec.maxdev, s = param.step, m = cSpec.avgdev)

    # Plot the intitial data.
    plotResults(itr, param, cSpec, logs)

    # Return the CalculatedSpectrum object and some data
    return time1, itr, cSpec

def timeString(x):
    ''' Formats the time given by the number x (in s), to a string that displays
        the time in HH:MM:SS.S format. Returns the formatted string.'''

    if not np.isfinite(x):
       return '???'

    h = x // 3600 # hours
    m = (x - (h * 3600)) // 60 # minutes
    s = np.floor(x - (h * 3600) - (m * 60)) # seconds
    d = np.floor((x - (h * 3600) - (m * 60) - s) * 10) # tenths of a second
    a = '%d.%1d' % (s, d) if x < 60 else '%02d.%1d' % (s, d)

    if m + h > 0:
        a = ('%d:' % m) + a if x < 3600 else ('%d:%02d:' % (h, m)) + a
    else:
        # Less than 1 minute, so just add 's'
        a += 's'

    # Return the formatted string
    return a

def writeData(labSpec, cSpec, param, bak = False, fname = None):
    ''' Writes (wavenumber, spectrum, n, k) data to the file named in parameter 
        fname, which defaults to the string stored in param.outputf. 

        Includes information from the param object in the header of the output 
        file. Nothing is returned. 

        OPTIONS:
            - If the keyword bak is True, the output file is "icenk-bak.tmp" (default)
        '''

    if fname is None:
        fname = param.outputf

    if bak:
        fname = param.bakf

    # Should the output be in CSV format?
    csv = fname.lower().endswith('csv')

    hedr = 'Output from ' + code_name + ' version ' + code_version + ' on %s\n' % time.asctime()

    if param.comment > '':
        hedr += '\n' + param.comment + '\n\n'

    hedr += 'Spectrum file          "%s"\n' % param.spectrumf
    hedr += 'Substrate n and k from "' + param.substratef + '"\n'
    hedr += 'Sample thickness       %.3f microns\n' % (1e+4*param.h)
    hedr += 'Sample visible index   %.3f at a wavelength of %.1f nm\n' % (param.nvis, param.laser * 1e+7)
    hedr += 'Input parameters from  "' + param.inputf + '"\n'
    hedr += '\n'
    hedr += 'If this code is used in your research, please cite '
    hedr += article_citation + '\n'
    hedr += '\n'
    hedr += 'For the original Python code, calculation details, and general information\n'
    hedr += 'from our group - please visit http://science.gsfc.nasa.gov/691/cosmicice\n'
    hedr += '\n'

    h1 = ['Wavnum [cm-1]', 'Abs(Calculated)',  '       n       ',  
          '       k       ']
    h2 = ['-------------', '---------------',  '---------------',
          '---------------']

    # Put delimiters into the column header rows and add them to the header string
    delim = '    '
    if csv:
        delim = ','

    for s in h1[:-1]:
        hedr += s + delim

    hedr += h1[-1] + '\n'


    for s in h2[:-1]:
        hedr += s + delim

    hedr += h2[-1] + '\n'

    # Write everything to the output file

    np.savetxt(fname, list(zip(cSpec.x, cSpec.y-cSpec.fringes, cSpec.n, cSpec.k)), 
        header = hedr, fmt = '%14.9e', delimiter = delim)

    return

# If running this as a script, read name of input file from command-line argument and
# begin calculations
if __name__ == "__main__":
    # Look for a command-line argument (the name of the input file).
    if len(sys.argv) == 1:
        # No argument was given, use the default filename
        infile = 'icenk-inputfile.txt'
    else:
        # Use the 1st argument from the command line
        infile = sys.argv[1]

    warnings.filterwarnings('ignore') # Hide warnings about division by 0, etc.

    findNK(infile) # run the main routine

    # wait for user interaction before quitting (in some cases, this prevents
    # the plot window from closing before the user can look at it)
    try:
        jnk = raw_input('\n> Hit ENTER to quit.')
    except:
        jnk = input('\n> Hit ENTER to quit.')

    printl('\n## END ## ' + code_name + ' v' + code_version + ' -- ' + time.asctime() + ' ##\n')

    

