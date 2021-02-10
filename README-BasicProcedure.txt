Instructions for using the icenk program

Note that all ASCII files are assumed to be whitespace-delimited, plain-text 
format, unless the file extension is given as .CSV (or .csv).

Basic procedure:

I. Prepare a laboratory infrared spectrum of a film with known thickness and 
visible-wavelength refractive index.  Measure the infrared spectrum and the 
thickness of a thin film in transmission mode on a substrate with known optical 
constants (or refractive index). The refractive index of the film material at 
visible energies must be known. Flatten the baseline to remove channel fringes 
and set the baseline to zero absorption (or to a transmittance of 1). Store the 
spectrum in a 2-column ASCII file (wavenumber, spectrum). See 
<example1-spectrum.txt> for an example. 

II. Prepare the optical constants of the substrate.  Store the optical 
constants of the substrate in a 3-column ASCII file (wavenumber, n, k).  See 
<kbr-nk.txt> and <csi-nk.txt> for examples for two commonly used IR-transparent 
substrates (KBr and CsI).

III. Prepare the input file.  Create an ASCII file with all input parameters 
and their values.  See the description of <example1-inputfile.txt> below. The 
default name for the input file is <icenk-inputfile.txt>, but can be specified 
if the code is run from the command line.

IV. Run the code.  In Windows 10: double-click on <icenk.exe>.  In Linux or 
other operating system where python3 is installed: from a command prompt, enter 
the directory where the code is found and type 'python icenk.py'.  The name of 
the input file can be added as a command-line argument, otherwise the default 
input filename <icenk-inputfile.txt> will be assumed.  By default, a window 
will open to display iteration progress in several plots. All messages written 
to the text window will be appended to the file <icenk.log>. 

V. Check the output.  If the algorithm converges on a solution, a 4-column 
plain ASCII file (wavenumber, calculated absorbance, n, k) is written with the 
results. See <example1-output.txt> for an example.

Format of the input file <example1-inputfile.txt>:

-- 
## Example input file with minimal input parameters
comment             Amorphous CH3OH at 10K, n=1.314, 4 fringes, measured on a KBr substrate
file_output         example1-output.txt
file_spectrum       example1-spectrum.txt
file_substrate      kbr-nk.txt
thickness_cm        1.02E-04
visible_index       1.314
goal                1.00E-05
--

General format of each line: Column 1 contains the name of a parameter, and 
column 2 contains a value for that parameter. The names and values used in 
<example1-inputfile.txt> are described below.  The file <README-inputfiles.txt> 
contains the complete list.  

##: any line that begins with "##" is ignored by the program

comment: the string in column 2 will be copied to the output file

file_output: the name of an ASCII file for output. 

file_spectrum:  the name of an ASCII file containing the laboratory absorbance 
    or transmittance spectrum from which the optical constants are to be 
    derived.

file_substrate: the name of an ASCII file containing the optical constants of 
    the substrate material.

thickness_cm: the thickness of the ice film, in units of centimeters (cm).

visible_index: the value of n at a higher energy than is present in the 
    spectrum, usually at a wavelength of a visible-light laser.

goal: the desired maximum fractional deviation between the laboratory input 
    spectrum and the spectrum that is calculated by the final values of n and k.
